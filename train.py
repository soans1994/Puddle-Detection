import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
import os
from datetime import datetime
from loss import combined_loss
from deeplabv3plus import MultiTaskDeepLab
from deeplabv3plus_improved import MultiTaskDeepLab
from loader import RoadDefectDataset
from augment import SmartAugmentRoadDataset


# In your training script, add this at the beginning
torch.autograd.set_detect_anomaly(True)

def train_model(model, train_loader, val_loader, device, config):
    # Initialize
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True
    )

    # Create log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(config['log_dir'], f"run_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    output_dir = os.path.join(config['output_dir'], f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    best_val_loss = float('inf')

    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")

        # Training phase
        model.train()
        train_loss = 0.0
        start_time = time.time()

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            # Move data to device
            inputs = batch['image'].to(device)
            targets = {
                'target': batch['target'].to(device),
                'road_mask': batch['road_mask'].to(device),
                'depth': batch['depth'].to(device),
                #'normal': batch['normal'].to(device),
                'camera_intrinsic': batch['camera_intrinsic'].to(device),
                'camera_extrinsic': batch['camera_extrinsic'].to(device)
            }
            occlusion = batch['occlusion'].to(device)

            # Forward pass - model now returns dict with all outputs
            outputs = model(inputs)

            # Compute loss
            loss_dict = combined_loss(
                pred=outputs,  # dict with 'defect', 'depth', 'normal', 'camera'
                target=targets,
                road_mask=batch['road_mask'].to(device),
                occlusion_mask=occlusion,
                weights=config['loss_weights']
            )
            loss = loss_dict['total']

            # Backward pass
            loss.backward()

            # Optional gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()

            # Log every N batches
            if batch_idx % config['log_interval'] == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")
                # Log individual loss components
                writer.add_scalar('BatchLoss/train_total', loss.item(), epoch * len(train_loader) + batch_idx)
                writer.add_scalar('BatchLoss/train_seg', loss_dict['segmentation'].item(),
                                  epoch * len(train_loader) + batch_idx)
                writer.add_scalar('BatchLoss/train_depth', loss_dict['depth'].item(),
                                  epoch * len(train_loader) + batch_idx)
                # writer.add_scalar('BatchLoss/train_normal', loss_dict['normal'].item(),
                #                   epoch * len(train_loader) + batch_idx)
                writer.add_scalar('BatchLoss/train_camera', loss_dict['camera'].item(),
                                  epoch * len(train_loader) + batch_idx)

        # Validation phase
        val_loss = 0.0
        val_loss_components = {
            'seg': 0.0,
            'depth': 0.0,
            #'normal': 0.0,
            'camera': 0.0
        }
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['image'].to(device)
                targets = {
                    'target': batch['target'].to(device),
                    'road_mask': batch['road_mask'].to(device),
                    'depth': batch['depth'].to(device),
                    #'normal': batch['normal'].to(device),
                    'camera_intrinsic': batch['camera_intrinsic'].to(device),
                    'camera_extrinsic': batch['camera_extrinsic'].to(device)
                }
                occlusion = batch['occlusion'].to(device)

                outputs = model(inputs)
                loss_dict = combined_loss(
                    pred=outputs,
                    target=targets,
                    road_mask=batch['road_mask'].to(device),
                    occlusion_mask=occlusion,
                    weights=config['loss_weights']
                )

                val_loss += loss_dict['total'].item()
                val_loss_components['seg'] += loss_dict['segmentation'].item()
                val_loss_components['depth'] += loss_dict['depth'].item()
                #val_loss_components['normal'] += loss_dict['normal'].item()
                val_loss_components['camera'] += loss_dict['camera'].item()

        # Calculate epoch metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        for k in val_loss_components:
            val_loss_components[k] /= len(val_loader)
        epoch_time = time.time() - start_time

        # Update scheduler
        scheduler.step(val_loss)

        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

        # Log individual validation components
        writer.add_scalar('Loss/val_seg', val_loss_components['seg'], epoch)
        writer.add_scalar('Loss/val_depth', val_loss_components['depth'], epoch)
        #writer.add_scalar('Loss/val_normal', val_loss_components['normal'], epoch)
        writer.add_scalar('Loss/val_camera', val_loss_components['camera'], epoch)

        # Print progress
        print(f"Epoch {epoch + 1} complete")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Components - Seg: {val_loss_components['seg']:.4f}")
        print(f"                - Depth: {val_loss_components['depth']:.4f}")
        #print(f"                - Normal: {val_loss_components['normal']:.4f}")
        print(f"                - Camera: {val_loss_components['camera']:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
            print("  Val loss improved ")
            print("  Saved new best model ")

        # Save checkpoint
        if (epoch + 1) % config['checkpoint_interval'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, os.path.join(log_dir, f'checkpoint_epoch{epoch + 1}.pth'))

    writer.close()
    return model


if __name__ == "__main__":
    # Configuration - updated with camera weight
    config = {
        'lr': 2e-4, # 1e-4 for bs 8, when bs double, lr double
        'epochs': 100,
        'batch_size': 16,
        'loss_weights': (1.5, 0.5, 0.1),  # seg, depth, camera #(1.0, 0.5, 0.1)
        'log_dir': './logs',
        'log_interval': 10,
        'output_dir': './output',
        'checkpoint_interval': 5,
        'num_workers': 4
    }

    # Initialize
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model - ensure it outputs camera parameters too
    model = MultiTaskDeepLab().to(device)

    # Create datasets and loaders
    base_dataset = RoadDefectDataset('D:/Road Dataset/dataset_root', split='train')
    # train_dataset = SmartAugmentRoadDataset(
    #     base_dataset,
    #     augment_percent=10.0 * 100.0,  # 30x augmentation  30.0 * 100.0
    #     augment_strength=0.3  # Moderate strength
    # )
    train_dataset = base_dataset
    # Visualize
    #train_dataset.visualize_augmentations(num_samples=5)

    val_dataset = RoadDefectDataset('D:/Road Dataset/dataset_root', split='val')

    # Check one sample
    sample = train_dataset[0]
    print(f"\nSample verification:")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Target shape: {sample['target'].shape}")
    print(f"Road mask shape: {sample['road_mask'].shape}")
    print(f"Camera intrinsic shape: {sample['camera_intrinsic'].shape}")
    print(f"Camera extrinsic shape: {sample['camera_extrinsic'].shape}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        collate_fn=RoadDefectDataset.collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        collate_fn=RoadDefectDataset.collate_fn
    )

    # Train
    trained_model = train_model(model, train_loader, val_loader, device, config)

    # Save final model
    torch.save(trained_model.state_dict(), os.path.join(config['output_dir'], 'final_model.pth'))