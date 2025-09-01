from pathlib import Path
import json
import logging
import shutil
from typing import List, Dict, Any, Set, Optional, Tuple
import pandas as pd
from dataclasses import dataclass
import traceback

try:
    import pyav
    PYAV_AVAILABLE = True
except ImportError:
    PYAV_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from .lerobot_dataset import LeRobotDatasetMetadata
from .compute_stats import aggregate_stats, compute_episode_stats
from .utils import (
    write_info, write_stats, write_jsonlines, serialize_dict,
    TASKS_PATH, EPISODES_PATH, EPISODES_STATS_PATH
)


@dataclass
class EpisodeValidationResult:
    """Results of validating a single episode."""
    episode_index: int
    is_valid: bool
    missing_data_file: bool = False
    corrupted_data_file: bool = False
    missing_video_files: List[str] = None
    corrupted_video_files: List[str] = None
    error_messages: List[str] = None
    
    def __post_init__(self):
        if self.missing_video_files is None:
            self.missing_video_files = []
        if self.corrupted_video_files is None:
            self.corrupted_video_files = []
        if self.error_messages is None:
            self.error_messages = []


@dataclass
class DatasetValidationReport:
    """Complete validation report for a dataset."""
    dataset_path: Path
    total_episodes: int
    valid_episodes: int
    invalid_episodes: List[EpisodeValidationResult]
    missing_episodes: List[int]
    validation_summary: Dict[str, int]
    
    def print_summary(self):
        """Print a human-readable summary of the validation."""
        print(f"\n{'='*60}")
        print(f"DATASET VALIDATION REPORT")
        print(f"{'='*60}")
        print(f"Dataset Path: {self.dataset_path}")
        print(f"Total Episodes in Metadata: {self.total_episodes}")
        print(f"Valid Episodes: {self.valid_episodes}")
        print(f"Invalid Episodes: {len(self.invalid_episodes)}")
        
        if self.missing_episodes:
            print(f"Missing Episodes: {len(self.missing_episodes)}")
            print(f"Missing Episode IDs: {sorted(self.missing_episodes)}")
        
        print(f"\nValidation Summary:")
        for issue_type, count in self.validation_summary.items():
            if count > 0:
                print(f"  - {issue_type}: {count}")
        
        if self.invalid_episodes:
            print(f"\nDetailed Issues:")
            for result in self.invalid_episodes:
                print(f"\n  Episode {result.episode_index}:")
                for msg in result.error_messages:
                    print(f"    - {msg}")


def validate_dataset(
    dataset_path: str | Path,
    check_video_integrity: bool = True,
    max_episodes_to_check: Optional[int] = None
) -> DatasetValidationReport:
    """
    Validate a dataset and identify corrupted or missing episodes.
    
    Args:
        dataset_path: Path to the dataset root directory
        check_video_integrity: Whether to perform deep video file validation
        max_episodes_to_check: Limit validation to first N episodes (for large datasets)
    
    Returns:
        DatasetValidationReport: Comprehensive validation results
    """
    dataset_path = Path(dataset_path)
    
    logging.info(f"Starting validation of dataset: {dataset_path}")
    
    # Load metadata
    try:
        # Extract repo_id from path name or use temporary one
        repo_id = dataset_path.name or "temp_validation"
        metadata = LeRobotDatasetMetadata(repo_id, root=dataset_path, allow_download=False)
    except Exception as e:
        raise ValueError(f"Failed to load dataset metadata: {e}")
    
    total_episodes = metadata.total_episodes
    episodes_to_check = list(range(total_episodes))
    
    if max_episodes_to_check:
        episodes_to_check = episodes_to_check[:max_episodes_to_check]
        logging.info(f"Limiting validation to first {max_episodes_to_check} episodes")
    
    # Validate each episode
    invalid_episodes = []
    missing_episodes = []
    validation_summary = {
        "missing_data_files": 0,
        "corrupted_data_files": 0,
        "missing_video_files": 0,
        "corrupted_video_files": 0,
        "metadata_inconsistencies": 0
    }
    
    for episode_idx in episodes_to_check:
        logging.debug(f"Validating episode {episode_idx}")
        
        # Check if episode exists in metadata
        if episode_idx not in metadata.episodes:
            missing_episodes.append(episode_idx)
            validation_summary["metadata_inconsistencies"] += 1
            continue
            
        result = _validate_episode(
            dataset_path, metadata, episode_idx, check_video_integrity
        )
        
        if not result.is_valid:
            invalid_episodes.append(result)
            
            # Update summary counts
            if result.missing_data_file:
                validation_summary["missing_data_files"] += 1
            if result.corrupted_data_file:
                validation_summary["corrupted_data_files"] += 1
            if result.missing_video_files:
                validation_summary["missing_video_files"] += len(result.missing_video_files)
            if result.corrupted_video_files:
                validation_summary["corrupted_video_files"] += len(result.corrupted_video_files)
    
    valid_episodes = len(episodes_to_check) - len(invalid_episodes) - len(missing_episodes)
    
    report = DatasetValidationReport(
        dataset_path=dataset_path,
        total_episodes=total_episodes,
        valid_episodes=valid_episodes,
        invalid_episodes=invalid_episodes,
        missing_episodes=missing_episodes,
        validation_summary=validation_summary
    )
    
    logging.info(f"Validation complete. {valid_episodes}/{len(episodes_to_check)} episodes are valid")
    return report


def _validate_episode(
    dataset_path: Path,
    metadata: LeRobotDatasetMetadata,
    episode_idx: int,
    check_video_integrity: bool
) -> EpisodeValidationResult:
    """Validate a single episode and return detailed results."""
    result = EpisodeValidationResult(episode_index=episode_idx, is_valid=True)
    
    # Check data file exists and is readable
    data_file_path = dataset_path / metadata.get_data_file_path(episode_idx)
    
    if not data_file_path.exists():
        result.missing_data_file = True
        result.is_valid = False
        result.error_messages.append(f"Missing data file: {data_file_path}")
    else:
        # Try to read parquet file
        try:
            df = pd.read_parquet(data_file_path)
            
            # Basic validation checks
            expected_length = metadata.episodes[episode_idx]["length"]
            if len(df) != expected_length:
                result.error_messages.append(
                    f"Data file length mismatch: found {len(df)}, expected {expected_length}"
                )
                result.is_valid = False
                
            # Check episode_index consistency
            if "episode_index" in df.columns:
                unique_indices = df["episode_index"].unique()
                if len(unique_indices) != 1 or unique_indices[0] != episode_idx:
                    result.error_messages.append(
                        f"Episode index mismatch in data: found {unique_indices}, expected [{episode_idx}]"
                    )
                    result.is_valid = False
                    
        except Exception as e:
            result.corrupted_data_file = True
            result.is_valid = False
            result.error_messages.append(f"Corrupted data file: {str(e)}")
    
    # Check video files if present
    for video_key in metadata.video_keys:
        video_file_path = dataset_path / metadata.get_video_file_path(episode_idx, video_key)
        
        if not video_file_path.exists():
            result.missing_video_files.append(video_key)
            result.is_valid = False
            result.error_messages.append(f"Missing video file: {video_file_path}")
        elif check_video_integrity:
            # Check video file integrity
            if not _validate_video_file(video_file_path):
                result.corrupted_video_files.append(video_key)
                result.is_valid = False
                result.error_messages.append(f"Corrupted video file: {video_file_path}")
    
    return result


def _validate_video_file(video_path: Path) -> bool:
    """Validate that a video file can be opened and read."""
    try:
        # Try with pyav first (more robust)
        if PYAV_AVAILABLE:
            with pyav.open(str(video_path)) as container:
                video_stream = container.streams.video[0]
                # Try to decode first frame
                for packet in container.demux(video_stream):
                    for frame in packet.decode():
                        return True
                        
        # Fallback to OpenCV
        elif CV2_AVAILABLE:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return False
            ret, frame = cap.read()
            cap.release()
            return ret
            
        else:
            # Basic file size check as last resort
            return video_path.stat().st_size > 0
            
    except Exception as e:
        logging.debug(f"Video validation failed for {video_path}: {e}")
        return False
    
    return False


def clean_dataset(
    dataset_path: str | Path,
    episodes_to_remove: List[int],
    output_path: Optional[str | Path] = None,
    output_repo_id: Optional[str] = None,
    backup_original: bool = True
) -> Path:
    """
    Clean a dataset by removing specified episodes.
    
    Args:
        dataset_path: Path to the input dataset
        episodes_to_remove: List of episode indices to remove
        output_path: Path for cleaned dataset (if None, modifies in place)
        output_repo_id: New repo_id for cleaned dataset
        backup_original: Create backup before modifying in-place
    
    Returns:
        Path: Path to the cleaned dataset
    """
    dataset_path = Path(dataset_path)
    
    # Determine output location
    if output_path is None:
        if backup_original:
            backup_path = dataset_path.parent / f"{dataset_path.name}_backup"
            logging.info(f"Creating backup at: {backup_path}")
            shutil.copytree(dataset_path, backup_path, dirs_exist_ok=True)
        output_path = dataset_path
    else:
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Copy dataset to new location first
        logging.info(f"Copying dataset to: {output_path}")
        shutil.copytree(dataset_path, output_path, dirs_exist_ok=True)
    
    # Load metadata
    repo_id = output_repo_id or dataset_path.name or "cleaned_dataset"
    metadata = LeRobotDatasetMetadata(repo_id, root=output_path, allow_download=False)
    
    episodes_to_remove = set(episodes_to_remove)
    logging.info(f"Removing {len(episodes_to_remove)} episodes: {sorted(episodes_to_remove)}")
    
    # Remove data files and update metadata
    _remove_episodes_and_update_metadata(output_path, metadata, episodes_to_remove, repo_id)
    
    logging.info(f"Dataset cleaning complete. Cleaned dataset saved to: {output_path}")
    return output_path


def _remove_episodes_and_update_metadata(
    dataset_path: Path,
    metadata: LeRobotDatasetMetadata,
    episodes_to_remove: Set[int],
    repo_id: str
) -> None:
    """Remove episodes and update all metadata accordingly."""
    
    # Remove data files
    for episode_idx in episodes_to_remove:
        if episode_idx in metadata.episodes:
            # Remove data file
            data_file = dataset_path / metadata.get_data_file_path(episode_idx)
            if data_file.exists():
                data_file.unlink()
                logging.debug(f"Removed data file: {data_file}")
            
            # Remove video files
            for video_key in metadata.video_keys:
                video_file = dataset_path / metadata.get_video_file_path(episode_idx, video_key)
                if video_file.exists():
                    video_file.unlink()
                    logging.debug(f"Removed video file: {video_file}")
    
    # Update metadata
    remaining_episodes = {
        idx: ep for idx, ep in metadata.episodes.items() 
        if idx not in episodes_to_remove
    }
    
    remaining_episodes_stats = {
        idx: stats for idx, stats in metadata.episodes_stats.items()
        if idx not in episodes_to_remove
    }
    
    # Recalculate statistics
    if remaining_episodes_stats:
        new_stats = aggregate_stats(list(remaining_episodes_stats.values()))
    else:
        new_stats = {}
    
    # Update info
    new_total_episodes = len(remaining_episodes)
    new_total_frames = sum(ep["length"] for ep in remaining_episodes.values())
    chunk_size = metadata.chunks_size
    new_total_chunks = (new_total_episodes - 1) // chunk_size + 1 if new_total_episodes > 0 else 0
    
    updated_info = metadata.info.copy()
    updated_info.update({
        "repo_id": repo_id,
        "total_episodes": new_total_episodes,
        "total_frames": new_total_frames,
        "total_chunks": new_total_chunks,
        "splits": {"train": f"0:{new_total_episodes}"},
        "total_videos": new_total_episodes * len(metadata.video_keys)
    })
    
    # Save updated metadata
    _save_cleaned_metadata(
        dataset_path, updated_info, metadata.tasks, remaining_episodes, 
        remaining_episodes_stats, new_stats
    )


def _save_cleaned_metadata(
    dataset_path: Path,
    info: Dict[str, Any],
    tasks: Dict[int, str],
    episodes: Dict[int, Dict],
    episodes_stats: Dict[int, Dict],
    stats: Dict[str, Any]
) -> None:
    """Save updated metadata after cleaning."""
    
    # Use existing utility functions for info and stats
    write_info(info, dataset_path)
    write_stats(stats, dataset_path)
    
    # Convert and write tasks
    tasks_list = [
        {"task_index": task_idx, "task": task}
        for task_idx, task in sorted(tasks.items())
    ]
    write_jsonlines(tasks_list, dataset_path / TASKS_PATH)
    
    # Write episodes (assuming they already have episode_index)
    episodes_list = [episodes[episode_idx] for episode_idx in sorted(episodes.keys())]
    write_jsonlines(episodes_list, dataset_path / EPISODES_PATH)
    
    # Write episodes_stats with proper serialization
    episodes_stats_list = [
        {"episode_index": episode_idx, "stats": serialize_dict(episode_stats)}
        for episode_idx, episode_stats in sorted(episodes_stats.items())
    ]
    write_jsonlines(episodes_stats_list, dataset_path / EPISODES_STATS_PATH)


def interactive_dataset_cleaning(
    dataset_path: str | Path,
    max_episodes_to_check: Optional[int] = None
) -> Optional[Path]:
    """
    Interactive dataset cleaning workflow with user prompts.
    
    Args:
        dataset_path: Path to the dataset to clean
        max_episodes_to_check: Limit validation to first N episodes
    
    Returns:
        Path to cleaned dataset or None if cancelled
    """
    dataset_path = Path(dataset_path)
    
    print(f"Starting interactive dataset cleaning for: {dataset_path}")
    print("Step 1: Validating dataset (dry run)...")
    
    # Validate dataset
    report = validate_dataset(
        dataset_path, 
        check_video_integrity=True,
        max_episodes_to_check=max_episodes_to_check
    )
    
    # Show report
    report.print_summary()
    
    if not report.invalid_episodes and not report.missing_episodes:
        print("\n✅ Dataset is clean! No issues found.")
        return None
    
    print(f"\n⚠️  Found issues in {len(report.invalid_episodes)} episodes.")
    
    # Ask user what to do
    print("\nOptions:")
    print("1. Remove all invalid episodes")
    print("2. Select specific episodes to remove")
    print("3. Cancel cleaning")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "3":
        print("Cleaning cancelled.")
        return None
    
    # Determine episodes to remove
    if choice == "1":
        episodes_to_remove = [ep.episode_index for ep in report.invalid_episodes]
        episodes_to_remove.extend(report.missing_episodes)
    elif choice == "2":
        print("\nInvalid episodes:")
        for i, ep in enumerate(report.invalid_episodes):
            print(f"  {i+1}. Episode {ep.episode_index} - {len(ep.error_messages)} issues")
        
        selections = input("\nEnter episode numbers to remove (comma-separated): ").strip()
        try:
            indices = [int(x.strip()) - 1 for x in selections.split(",")]
            episodes_to_remove = [report.invalid_episodes[i].episode_index for i in indices]
        except (ValueError, IndexError):
            print("Invalid selection. Cleaning cancelled.")
            return None
    else:
        print("Invalid choice. Cleaning cancelled.")
        return None
    
    if not episodes_to_remove:
        print("No episodes selected for removal.")
        return None
    
    print(f"\nEpisodes to remove: {sorted(episodes_to_remove)}")
    
    # Ask about output location
    in_place = input("\nModify dataset in place? (y/N): ").lower().startswith('y')
    
    if in_place:
        backup = input("Create backup before modifying? (Y/n): ").lower() != 'n'
        output_path = None
        output_repo_id = None
    else:
        output_path = input("Enter path for cleaned dataset: ").strip()
        output_repo_id = input("Enter new repo_id (optional): ").strip() or None
        backup = False
    
    # Confirm cleaning
    print(f"\nAbout to remove {len(episodes_to_remove)} episodes...")
    confirm = input("Continue? (y/N): ").lower().startswith('y')
    
    if not confirm:
        print("Cleaning cancelled.")
        return None
    
    # Perform cleaning
    print("\nStep 2: Cleaning dataset...")
    cleaned_path = clean_dataset(
        dataset_path=dataset_path,
        episodes_to_remove=episodes_to_remove,
        output_path=output_path,
        output_repo_id=output_repo_id,
        backup_original=backup
    )
    
    print(f"\n✅ Dataset cleaning complete!")
    print(f"Cleaned dataset location: {cleaned_path}")
    
    return cleaned_path
