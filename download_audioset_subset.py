"""
Download AudioSet subset for Music Genre Classification.
Filters for music-related categories and downloads audio from YouTube.
"""
import os
import pandas as pd
import subprocess
from collections import defaultdict

# AudioSet music categories (ontology IDs)
MUSIC_CATEGORIES = {
    '/m/04rlf': 'Music',
    '/m/0glt670': 'Electronic',
    '/m/02lkt': 'Rock',
    '/m/0gywn': 'Pop',
    '/m/03_d0': 'Jazz',
    '/m/0342h': 'Hip-Hop',
    '/m/015lz1': 'Indie',
    '/m/07gxw': 'Techno',
    '/m/0dl5d': 'Drum_and_Bass',
    '/m/05w3f': 'Folk',
    '/m/06cqb': 'Reggae',
    '/m/0y4f8': 'Classical',
}

AUDIOSET_CSV_URL = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv"
OUTPUT_DIR = os.path.join('.', 'datasets', 'audioset_music')
MAX_PER_GENRE = 200  # Download 200 samples per genre

os.makedirs(OUTPUT_DIR, exist_ok=True)


def download_audioset_csv():
    """Download AudioSet CSV metadata."""
    print('Downloading AudioSet metadata...')
    csv_path = os.path.join(OUTPUT_DIR, 'audioset_metadata.csv')
    
    if os.path.exists(csv_path):
        print(f'Metadata already exists: {csv_path}')
        return csv_path
    
    # Download using curl or wget
    cmd = f'curl -o "{csv_path}" {AUDIOSET_CSV_URL}'
    subprocess.run(cmd, shell=True)
    print(f'Downloaded to: {csv_path}')
    return csv_path


def filter_music_clips(csv_path):
    """Filter AudioSet CSV for music-related clips."""
    print('\nFiltering for music categories...')
    
    # Read CSV (skip first 3 header lines)
    df = pd.read_csv(csv_path, skiprows=3, names=['YTID', 'start_seconds', 'end_seconds', 'positive_labels'])
    
    # Filter by music categories
    music_clips = defaultdict(list)
    
    for idx, row in df.iterrows():
        labels = row['positive_labels'].split(',')
        
        for label in labels:
            label = label.strip().strip('"')
            if label in MUSIC_CATEGORIES:
                genre = MUSIC_CATEGORIES[label]
                if len(music_clips[genre]) < MAX_PER_GENRE:
                    music_clips[genre].append({
                        'ytid': row['YTID'],
                        'start': row['start_seconds'],
                        'end': row['end_seconds'],
                        'genre': genre
                    })
    
    # Print stats
    print(f'\nFiltered clips per genre:')
    for genre, clips in music_clips.items():
        print(f'  {genre:15s}: {len(clips):4d} clips')
    
    return music_clips


def download_audio_clips(music_clips):
    """Download filtered audio clips using yt-dlp."""
    print('\n' + '='*70)
    print('DOWNLOADING AUDIO CLIPS')
    print('='*70)
    print('Note: Requires yt-dlp installed: pip install yt-dlp')
    
    for genre, clips in music_clips.items():
        genre_dir = os.path.join(OUTPUT_DIR, genre)
        os.makedirs(genre_dir, exist_ok=True)
        
        print(f'\nDownloading {genre} clips...')
        
        for i, clip in enumerate(clips):
            output_file = os.path.join(genre_dir, f"{clip['ytid']}_{int(clip['start'])}_{int(clip['end'])}.wav")
            
            if os.path.exists(output_file):
                continue
            
            # YouTube URL
            url = f"https://www.youtube.com/watch?v={clip['ytid']}"
            
            # yt-dlp command to download segment
            cmd = [
                'yt-dlp',
                '-f', 'bestaudio',
                '--extract-audio',
                '--audio-format', 'wav',
                '--audio-quality', '0',
                '--postprocessor-args', f"-ss {clip['start']} -t {clip['end'] - clip['start']}",
                '-o', output_file,
                url
            ]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True, timeout=30)
                print(f'  [{i+1}/{len(clips)}] Downloaded: {clip["ytid"]}')
            except:
                print(f'  [{i+1}/{len(clips)}] Failed: {clip["ytid"]}')
    
    print(f'\n✓ Download complete! Audio saved to: {OUTPUT_DIR}')


def create_metadata_csv(music_clips):
    """Create metadata CSV for downloaded clips."""
    metadata = []
    
    for genre, clips in music_clips.items():
        for clip in clips:
            metadata.append({
                'filename': f"{clip['ytid']}_{int(clip['start'])}_{int(clip['end'])}.wav",
                'genre': genre,
                'ytid': clip['ytid'],
                'start': clip['start'],
                'end': clip['end']
            })
    
    df = pd.DataFrame(metadata)
    csv_path = os.path.join(OUTPUT_DIR, 'audioset_music_metadata.csv')
    df.to_csv(csv_path, index=False)
    print(f'\n✓ Metadata saved to: {csv_path}')


if __name__ == '__main__':
    print('='*70)
    print('AUDIOSET MUSIC SUBSET DOWNLOADER')
    print('='*70)
    
    # Step 1: Download metadata
    csv_path = download_audioset_csv()
    
    # Step 2: Filter for music
    music_clips = filter_music_clips(csv_path)
    
    # Step 3: Create metadata
    create_metadata_csv(music_clips)
    
    # Step 4: Download audio (optional - can be slow)
    download = input('\nDownload audio clips now? (y/n): ').lower()
    if download == 'y':
        print('\nInstalling yt-dlp if needed...')
        subprocess.run('pip install yt-dlp', shell=True)
        download_audio_clips(music_clips)
    else:
        print('\nSkipped download. Run script again to download later.')
        print(f'Metadata saved to: {OUTPUT_DIR}')
