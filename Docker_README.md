# Docker Setup for Football AI

## Prerequisites

- Docker and Docker Compose installed
- NVIDIA Docker runtime (for GPU support)
- X11 forwarding enabled (for GUI display)

## Quick Start

1. Build and run the container:
```bash
docker-compose up -d
docker-compose exec football-ai bash
```

2. Once inside the container, run the Football AI:
```bash
python main.py --config config.yaml
```

## GPU Support

If you have NVIDIA GPU:
```bash
docker-compose -f docker-compose.gpu.yaml up -d
docker-compose -f docker-compose.gpu.yaml exec football-ai bash
```

## Using the Convenience Script

Make the script executable:
```bash
chmod +x run_docker.sh
```

Run:
```bash
./run_docker.sh
```

## Directory Structure

- `/app` - Your code (mounted from current directory)
- `/app/videos` - Input videos (includes downloaded samples)
- `/app/output` - Output videos
- `/app/.cache` - Model cache

## Sample Videos

The following sample videos are automatically downloaded:
- 0bfacc_0.mp4
- 2e57b9_0.mp4
- 08fd33_0.mp4
- 573e61_0.mp4
- 121364_0.mp4

## Updating Config

To use a sample video, update your `config.yaml`:
```yaml
video:
  input_path: "/app/videos/08fd33_0.mp4"
  output_path: "/app/output/output_video.mp4"
```

## Troubleshooting

### GUI Display Issues
If you get display errors:
```bash
# On host machine
xhost +local:docker
```

### Permission Issues
If you encounter permission issues with output files:
```bash
# Inside container
chmod -R 777 /app/output
```

### GPU Not Detected
Make sure you have:
- NVIDIA drivers installed
- NVIDIA Docker runtime installed
- Using the GPU compose file

## Notes

- The container runs interactively, allowing you to modify code and rerun
- All changes to code are reflected immediately (volume mounted)
- Output videos are saved to the `output` folder
- Model caches are preserved between runs
