import argparse
import copy
import datetime
import inspect
import json
import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path

from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma

from datasets import build_dataset
from engine import train_one_epoch, evaluate
from main import get_args_parser
from models import *
from optim_factory import LayerDecayValueAssigner, create_optimizer
from utils import NativeScalerWithGradNormCount as NativeScaler
import utils


def build_qat_parser():
    parser = argparse.ArgumentParser(
        'QAT fine-tuning script for image classification',
        parents=[get_args_parser()],
    )
    parser.add_argument(
        '--qat_checkpoint',
        default='runs/checkpoint-best.pth',
        type=str,
        help='Path to the pretrained .pth checkpoint that will be loaded before QAT starts',
    )
    parser.add_argument(
        '--qat_quantizer',
        default='int8_dynamic_act_int4_weight',
        choices=['int8_dynamic_act_int4_weight', 'int4_weight_only'],
        help='TorchAO QAT quantizer preset',
    )
    parser.add_argument('--qat_group_size', default=32, type=int, help='Group size for supported int4 quantizers')
    parser.add_argument('--qat_inner_k_tiles', default=8, type=int, help='Inner K tiles for supported int4 quantizers')
    parser.add_argument(
        '--qat_save_converted',
        type=str2bool,
        default=True,
        help='Convert the trained QAT model to a quantized model and save an extra checkpoint',
    )
    parser.set_defaults(
        use_amp=False,
        output_dir='runs_qat',
        lr=1e-5,
        min_lr=1e-5,
        warmup_epochs=0,
        warmup_steps=0,
        weight_decay=1e-5,
        weight_decay_end=1e-5,
        epochs=10,
        mixup=0.0,
        cutmix=0.0,
        smoothing=0.0,
        model_ema=False,
        model_ema_eval=False,
        auto_resume=False,
    )
    return parser


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def create_model(args):
    if args.model == "mobilenet_v3_small":
        return MobileNetV3_Small(num_classes=args.nb_classes)
    if args.model == "mobilenet_v3_large":
        return MobileNetV3_Large(num_classes=args.nb_classes)
    if args.model == "senet18":
        return se_resnet_18(num_classes=args.nb_classes)
    if args.model == "senet34":
        return se_resnet_34(num_classes=args.nb_classes)
    raise ValueError(f"Unsupported model: {args.model}")


def load_checkpoint_model(model, checkpoint_path, model_key, model_prefix):
    checkpoint = utils.load_checkpoint(checkpoint_path, map_location='cpu')
    checkpoint_model = None
    for key in model_key.split('|'):
        if key in checkpoint:
            checkpoint_model = checkpoint[key]
            print(f"Load state_dict by model_key = {key}")
            break
    if checkpoint_model is None:
        checkpoint_model = checkpoint

    state_dict = model.state_dict()
    classifier_keys = [
        'head.weight', 'head.bias',
        'fc.weight', 'fc.bias',
        'linear4.weight', 'linear4.bias',
    ]
    for key in classifier_keys:
        if key in checkpoint_model and key in state_dict and checkpoint_model[key].shape != state_dict[key].shape:
            print(f"Removing key {key} from pretrained checkpoint due to shape mismatch")
            del checkpoint_model[key]

    utils.load_state_dict(model, checkpoint_model, prefix=model_prefix)


def resolve_qat_quantizer_cls(name):
    candidates = [
        "torchao.quantization.prototype.qat",
        "torchao.quantization.qat",
        "torchao.prototype.qat",
    ]
    class_names = {
        "int8_dynamic_act_int4_weight": "Int8DynActInt4WeightQATQuantizer",
        "int4_weight_only": "Int4WeightOnlyQATQuantizer",
    }

    errors = []
    for module_name in candidates:
        try:
            module = __import__(module_name, fromlist=[class_names[name]])
        except Exception as exc:
            errors.append(f"{module_name}: {exc}")
            continue

        quantizer_cls = getattr(module, class_names[name], None)
        if quantizer_cls is not None:
            return quantizer_cls

    raise ImportError(
        "Unable to locate a TorchAO QAT quantizer class. Tried: "
        + "; ".join(errors)
    )


def build_quantizer(args):
    quantizer_cls = resolve_qat_quantizer_cls(args.qat_quantizer)
    signature = inspect.signature(quantizer_cls)
    kwargs = {}
    if 'group_size' in signature.parameters:
        kwargs['group_size'] = args.qat_group_size
    if 'groupsize' in signature.parameters:
        kwargs['groupsize'] = args.qat_group_size
    if 'inner_k_tiles' in signature.parameters:
        kwargs['inner_k_tiles'] = args.qat_inner_k_tiles
    return quantizer_cls(**kwargs)


def prepare_model_for_qat(model, args):
    quantizer = build_quantizer(args)
    prepared_model = quantizer.prepare(model)
    if prepared_model is not None:
        model = prepared_model
    return model, quantizer


def convert_qat_model(model, quantizer):
    converted_model = quantizer.convert(model)
    if converted_model is not None:
        model = converted_model
    return model


def save_converted_model(args, model_without_ddp, quantizer, epoch_label):
    if not args.output_dir or not utils.is_main_process():
        return

    quantized_model = copy.deepcopy(model_without_ddp)
    quantized_model.eval()
    converted_model = convert_qat_model(quantized_model, quantizer)
    save_path = Path(args.output_dir) / f'checkpoint-{epoch_label}-converted.pth'
    torch.save(
        {
            'model': converted_model.state_dict(),
            'epoch': epoch_label,
            'args': args,
            'qat_quantizer': args.qat_quantizer,
        },
        save_path,
    )
    print(f"Saved converted QAT checkpoint to {save_path}")


def main(args):
    utils.init_distributed_mode(args)
    print(args)
    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    if args.disable_eval:
        args.dist_eval = False
        dataset_val = None
        sampler_val = None
    else:
        dataset_val, _ = build_dataset(is_train=False, args=args)
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=args.seed,
    )
    print("Sampler_train = %s" % str(sampler_train))

    if global_rank == 0 and args.enable_wandb:
        wandb_logger = utils.WandbLogger(args)
    else:
        wandb_logger = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=125,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )
    else:
        data_loader_val = None

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    model = create_model(args)
    print(f"Loading pretrained checkpoint from {args.qat_checkpoint}")
    load_checkpoint_model(model, args.qat_checkpoint, args.model_key, args.model_prefix)
    model, qat_quantizer = prepare_model_for_qat(model, args)
    model.to(device)

    model_ema = None
    if args.model_ema:
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)

    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)

    if args.layer_decay < 1.0 or args.layer_decay > 1.0:
        num_layers = 12
        assert args.model in ['convnext_small', 'convnext_base', 'convnext_large', 'convnext_xlarge'], \
            "Layer Decay impl only supports convnext_small/base/large/xlarge"
        assigner = LayerDecayValueAssigner(
            list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2))
        )
    else:
        assigner = None

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    optimizer = create_optimizer(
        args, model_without_ddp, skip_list=None,
        get_num_layer=assigner.get_layer_id if assigner is not None else None,
        get_layer_scale=assigner.get_scale if assigner is not None else None)

    loss_scaler = NativeScaler()

    print("Use constant LR / WD for QAT fine-tuning")
    lr_schedule_values = None
    wd_schedule_values = None
    for param_group in optimizer.param_groups:
        param_group["lr"] = args.lr * param_group.get("lr_scale", 1.0)
        if param_group["weight_decay"] > 0:
            param_group["weight_decay"] = args.weight_decay
    print("Constant LR = %.8f" % args.lr)
    print("Constant WD = %.8f" % args.weight_decay)

    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

    if args.eval:
        print("Eval only mode")
        test_stats = evaluate(data_loader_val, model, device, use_amp=args.use_amp)
        print(f"Accuracy of the network on {len(dataset_val)} test images: {test_stats['acc1']:.5f}%")
        if args.qat_save_converted:
            save_converted_model(args, model_without_ddp, qat_quantizer, "eval")
        return

    max_accuracy = 0.0
    if args.model_ema and args.model_ema_eval:
        max_accuracy_ema = 0.0

    print("Start QAT training for %d epochs" % args.epochs)
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if wandb_logger:
            wandb_logger.set_steps()

        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer,
            device, epoch, loss_scaler, args.clip_grad, model_ema, mixup_fn,
            log_writer=None, wandb_logger=wandb_logger, start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq,
            use_amp=args.use_amp
        )

        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema)

        if data_loader_val is not None:
            test_stats = evaluate(data_loader_val, model, device, use_amp=args.use_amp)
            print(f"Accuracy of the model on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            if max_accuracy < test_stats["acc1"]:
                max_accuracy = test_stats["acc1"]
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)
                    if args.qat_save_converted:
                        save_converted_model(args, model_without_ddp, qat_quantizer, "best")
            print(f'Max accuracy: {max_accuracy:.2f}%')

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

            if args.model_ema and args.model_ema_eval:
                test_stats_ema = evaluate(data_loader_val, model_ema.ema, device, use_amp=args.use_amp)
                print(f"Accuracy of the model EMA on {len(dataset_val)} test images: {test_stats_ema['acc1']:.1f}%")
                if max_accuracy_ema < test_stats_ema["acc1"]:
                    max_accuracy_ema = test_stats_ema["acc1"]
                    if args.output_dir and args.save_ckpt:
                        utils.save_model(
                            args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch="best-ema", model_ema=model_ema)
                log_stats.update({**{f'test_{k}_ema': v for k, v in test_stats_ema.items()}})
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        if wandb_logger:
            wandb_logger.log_epoch_metrics(log_stats)

    if args.qat_save_converted:
        save_converted_model(args, model_without_ddp, qat_quantizer, "last")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = build_qat_parser()
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
