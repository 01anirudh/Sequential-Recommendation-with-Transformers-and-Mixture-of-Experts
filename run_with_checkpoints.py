import argparse
from logging import getLogger
import torch
import os
from time import time
from datetime import datetime
from recbole.config import Config
from recbole.data import data_preparation
from recbole.utils import init_seed, init_logger, set_color, get_trainer, early_stopping

from utils import get_model, create_dataset


def save_checkpoint(trainer, epoch, checkpoint_dir):
    """Save DUAL checkpoints: current.pth + previous.pth (bulletproof!)"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': trainer.model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'best_valid_score': trainer.best_valid_score,
        'cur_step': trainer.cur_step,
        'model_name': trainer.model.__class__.__name__,
        'save_timestamp': time(),
    }
    
    # Dual checkpoint paths
    current_path = os.path.join(checkpoint_dir, 'current.pth')
    previous_path = os.path.join(checkpoint_dir, 'previous.pth')
    current_tmp = os.path.join(checkpoint_dir, 'current.pth.tmp')
    
    try:
        # 1. Save NEW checkpoint (atomic write)
        torch.save(checkpoint, current_tmp)
        
        # 2. VALIDATE the saved checkpoint
        test_ckpt = torch.load(current_tmp, map_location='cpu')
        assert test_ckpt['epoch'] == epoch, f"Epoch mismatch: saved {test_ckpt['epoch']} != {epoch}"
        assert 'model_state_dict' in test_ckpt
        
        # 3. Atomic replace current.pth
        if os.path.exists(current_path):
            os.replace(current_tmp, current_path)
        else:
            os.rename(current_tmp, current_path)
        
        # 4. Update previous.pth (old current becomes previous)
        if os.path.exists(previous_path):
            os.remove(previous_path)
        if os.path.exists(current_path):
            os.rename(current_path, previous_path)
        
        print(f"✅ DUAL SAVE epoch {epoch}: current.pth + previous.pth")
        print(f"   📁 {checkpoint_dir}")
        return current_path
        
    except Exception as e:
        print(f"❌ SAVE FAILED epoch {epoch}: {e}")
        # Cleanup temp file
        if os.path.exists(current_tmp):
            try: os.remove(current_tmp)
            except: pass
        raise e


def load_checkpoint_safe(trainer, checkpoint_dir, config, logger):
    """Load: current.pth → previous.pth → epoch 0 (never lose >1 epoch!)"""
    paths = [
        os.path.join(checkpoint_dir, 'current.pth'),
        os.path.join(checkpoint_dir, 'previous.pth')
    ]
    
    for i, ckpt_path in enumerate(paths, 1):
        if not os.path.exists(ckpt_path): 
            continue
            
        logger.info(f"📥 [{i}/2] Loading: {os.path.basename(ckpt_path)}")
        logger.info(f"   📁 {ckpt_path}")
        
        try:
            ckpt = torch.load(ckpt_path, map_location=config['device'])
            
            # Validate checkpoint integrity
            required = ['model_state_dict', 'optimizer_state_dict', 'epoch']
            missing = [k for k in required if k not in ckpt]
            if missing:
                raise ValueError(f"Missing keys: {missing}")
            
            # Model compatibility check
            if 'model_name' in ckpt and ckpt['model_name'] != trainer.model.__class__.__name__:
                logger.warning(f"⚠️  Model mismatch: {ckpt['model_name']} → {trainer.model.__class__.__name__}")
            
            # Load states
            trainer.model.load_state_dict(ckpt['model_state_dict'])
            trainer.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            trainer.start_epoch = ckpt['epoch']
            trainer.best_valid_score = ckpt.get('best_valid_score', 0.0)
            trainer.cur_step = ckpt.get('cur_step', 0)
            
            # Success message with timestamp
            ts_str = ""
            if 'save_timestamp' in ckpt:
                ts = datetime.fromtimestamp(ckpt['save_timestamp']).strftime('%H:%M:%S')
                ts_str = f" (saved {ts})"
            
            logger.info(f"✅ RESUMED epoch {trainer.start_epoch}{ts_str}")
            logger.info(f"   Best score: {trainer.best_valid_score:.4f}")
            return True
            
        except Exception as e:
            logger.error(f"❌ [{i}/2] LOAD FAILED: {e}")
            continue
    
    logger.warning("⚠️  BOTH checkpoints failed. Starting epoch 0")
    trainer.start_epoch = 0
    return False


def manual_training_loop(trainer, train_data, valid_data, config, checkpoint_dir, checkpoint_freq):
    """Manual training loop with robust checkpointing"""
    logger = trainer.logger
    
    for epoch_idx in range(trainer.start_epoch, trainer.epochs):
        # Train epoch
        t0 = time()
        train_loss = trainer._train_epoch(train_data, epoch_idx, show_progress=config['show_progress'])
        trainer.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
        
        # Log training
        train_loss_output = trainer._generate_train_loss_output(epoch_idx, t0, time(), train_loss)
        logger.info(train_loss_output)
        trainer._add_train_loss_to_tensorboard(epoch_idx, train_loss)
        
        # SAVE CHECKPOINT (CRITICAL!)
        try:
            save_checkpoint(trainer, epoch_idx + 1, checkpoint_dir)
        except Exception as e:
            logger.error(f"❌ CHECKPOINT FAILED: {e}")
            logger.warning("⚠️  Continuing without checkpoint...")
        
        # Validate
        if trainer.eval_step <= 0 or not valid_data:
            if config['saved']:
                trainer._save_checkpoint(epoch_idx)
            continue
        
        if (epoch_idx + 1) % trainer.eval_step == 0:
            valid_start = time()
            valid_score, valid_result = trainer._valid_epoch(valid_data, show_progress=config['show_progress'])
            
            trainer.best_valid_score, trainer.cur_step, stop_flag, update_flag = early_stopping(
                valid_score, trainer.best_valid_score, trainer.cur_step,
                max_step=trainer.stopping_step, bigger=trainer.valid_metric_bigger
            )
            
            # Log validation
            logger.info(set_color(f"epoch {epoch_idx} evaluating", 'green') + 
                       f" [time: {time()-valid_start:.2f}s, valid_score: {valid_score:.6f}]")
            logger.info(set_color('valid result', 'blue') + f': {valid_result}')
            
            trainer.tensorboard.add_scalar('Valid_score', valid_score, epoch_idx)
            
            if update_flag:
                trainer._save_checkpoint(epoch_idx)
                trainer.best_valid_result = valid_result
                logger.info(set_color('Saving current best', 'blue') + f': {trainer.saved_model_file}')
            
            if stop_flag:
                logger.info(f'🎉 Finished training, best eval result in epoch {epoch_idx - trainer.cur_step * trainer.eval_step}')
                break
    
    trainer._add_hparam_to_tensorboard(trainer.best_valid_score)
    return trainer.best_valid_score, trainer.best_valid_result


def run_single(model_name, dataset, pretrained_file='', resume='', checkpoint_freq=5, 
               checkpoint_dir=None, skip_initial_checkpoint=False, **kwargs):
    
    # Config setup
    props = ['config/overall.yaml', f'config/{model_name}.yaml']
    print(f"📋 Config files: {props}")
    
    model_class = get_model(model_name)
    dataset_name = dataset
    
    config = Config(model=model_class, dataset=dataset, config_file_list=props, config_dict=kwargs)
    init_seed(config['seed'], config['reproducibility'])
    
    # Logger
    init_logger(config)
    logger = getLogger()
    logger.info(config)
    
    # Dataset
    dataset_obj = create_dataset(config)
    logger.info(dataset_obj)
    train_data, valid_data, test_data = data_preparation(config, dataset_obj)
    
    # Model
    model = model_class(config, train_data.dataset).to(config['device'])
    
    if pretrained_file:
        ckpt = torch.load(pretrained_file, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'], strict=False)
        logger.info(f'✅ Loaded pretrained: {pretrained_file}')
    
    logger.info(model)
    
    # Trainer
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    os.makedirs('saved', exist_ok=True)  # RecBole needs this
    
    # ═══ CHECKPOINT SETUP ═══
    if checkpoint_dir is None:
        checkpoint_dir = f"checkpoints/{model_name}_{dataset_name}"
    else:
        checkpoint_dir = os.path.join(checkpoint_dir, f"{model_name}_{dataset_name}")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger.info(f"💾 Checkpoints: {checkpoint_dir}")
    
    # RESUME TRAINING
    resume_success = False
    if resume == 'auto' or resume:
        resume_success = load_checkpoint_safe(trainer, checkpoint_dir, config, logger)
    
    # Initial checkpoint
    if not skip_initial_checkpoint and not resume_success:
        logger.info("💾 Saving initial checkpoint (epoch 0)...")
        try:
            save_checkpoint(trainer, 0, checkpoint_dir)
        except Exception as e:
            logger.warning(f"⚠️ Initial checkpoint failed: {e}")
    
    logger.info("🚀 Starting training...")
    
    # TRAINING LOOP
    try:
        best_valid_score, best_valid_result = manual_training_loop(
            trainer, train_data, valid_data, config, checkpoint_dir, checkpoint_freq
        )
    except KeyboardInterrupt:
        logger.warning("⛔ Training interrupted! Saving final checkpoint...")
        save_checkpoint(trainer, trainer.start_epoch, checkpoint_dir)
        logger.info("✅ Resume with: --resume auto")
        raise
    except Exception as e:
        logger.error(f"💥 Training crashed: {e}")
        save_checkpoint(trainer, trainer.start_epoch, checkpoint_dir)
        raise
    
    # FINAL EVALUATION (paper results!)
    logger.info("🏆 Final test evaluation...")
    try:
        test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=config['show_progress'])
        logger.info(f"✅ Test results (BEST model): {test_result}")
    except FileNotFoundError:
        logger.warning("⚠️ RecBole best model missing, using latest checkpoint...")
        ckpt_path = os.path.join(checkpoint_dir, 'current.pth')
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=config['device'])
            trainer.model.load_state_dict(ckpt['model_state_dict'])
        test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=config['show_progress'])

    # ── PER-USER NDCG@10 SAVE (for exact p-value computation) ────
    try:
        import math, numpy as np
        trainer.model.eval()
        per_user_ndcg = []

        with torch.no_grad():
            for batch in test_data:
                # RecBole FullSortEvalDataLoader yields (interaction, ...) tuples
                interaction = batch[0] if isinstance(batch, tuple) else batch
                interaction = interaction.to(config['device'])
                scores = trainer.model.full_sort_predict(interaction)   # (B, n_items)
                pos_items = interaction[trainer.model.POS_ITEM_ID]      # (B,)
                sorted_idx = torch.argsort(scores, dim=1, descending=True)

                for b in range(pos_items.shape[0]):
                    pid = pos_items[b].item()
                    hits = (sorted_idx[b] == pid).nonzero(as_tuple=True)[0]
                    if len(hits) == 0:
                        per_user_ndcg.append(0.0)
                    else:
                        rank = hits[0].item() + 1          # 1-indexed
                        per_user_ndcg.append(1.0 / math.log2(rank + 1) if rank <= 10 else 0.0)

        per_user_ndcg = np.array(per_user_ndcg)
        scores_dir = "pvalue_scores"
        os.makedirs(scores_dir, exist_ok=True)
        # Use only the top-level checkpoint dir (not the extended subpath)
        suffix = checkpoint_dir.split('/')[0].split('\\')[0] if checkpoint_dir else config['model']
        suffix = suffix.replace("/", "_").replace("\\", "_")
        score_path = os.path.join(scores_dir, f"{suffix}_{config['dataset']}_ndcg10.npy")
        np.save(score_path, per_user_ndcg)
        logger.info(f"📁 Per-user NDCG@10 saved → {score_path}  (n={len(per_user_ndcg)}, mean={per_user_ndcg.mean():.4f})")
    except Exception as e:
        import traceback
        logger.warning(f"⚠️  Could not save per-user NDCG scores: {e}")
        logger.warning(traceback.format_exc())

    # PAPER RESULTS
    logger.info("="*80)
    logger.info(set_color('🏆 BEST VALIDATION', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('📊 FINAL TEST', 'yellow') + f': {test_result}')
    logger.info("="*80)

    return config['model'], config['dataset'], {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=str, default='UniSRec', help='model name')
    parser.add_argument('-d', type=str, default='All_Beauty', help='dataset name')
    parser.add_argument('-p', type=str, default='', help='pretrained model')
    parser.add_argument('--resume', type=str, default='', help='resume (auto/current.pth/previous.pth)')
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='custom checkpoint dir')
    parser.add_argument('--skip-initial-checkpoint', action='store_true')
    args, unparsed = parser.parse_known_args()
    print("Parsed Args:", args)
    
    # Extract custom ablation flags (e.g. --use_deep_moe=False)
    custom_kwargs = {}
    for arg in unparsed:
        if arg.startswith('--') and '=' in arg:
            key, val = arg[2:].split('=', 1)
            custom_kwargs[key] = val
            
    print("Custom Kwargs correctly captured:", custom_kwargs)
    
    run_single(
        args.m, args.d, args.p, args.resume,
        checkpoint_dir=args.checkpoint_dir,
        skip_initial_checkpoint=args.skip_initial_checkpoint,
        **custom_kwargs
    )
