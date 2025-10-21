# roop/core.py

import os
import sys
import argparse
import onnxruntime
import cv2
import roop.globals
import roop.utilities as util
from settings import Settings
from roop.face_util import extract_face_images
from roop.FaceSet import FaceSet
from roop.ProcessOptions import ProcessOptions
from roop.ProcessMgr import ProcessMgr
from roop.capturer import get_image_frame, get_video_frame_total
from roop.util_ffmpeg import restore_audio

def parse_args():
    """Parses command-line arguments for the CLI."""
    parser = argparse.ArgumentParser(description="Roop Unleashed CLI")
    parser.add_argument("-s", "--source", required=True, help="Source image path with the face to use.")
    parser.add_argument("-t", "--target", required=True, help="Target image or video path to apply the face to.")
    parser.add_argument("-o", "--output", required=True, help="Path for the output file.")
    parser.add_argument(
        "--swap-model",
        choices=["InSwapper 128", "ReSwapper 128", "ReSwapper 256"],
        default="InSwapper 128",
        help="Face swap model to use.",
    )
    parser.add_argument(
        "--enhancer",
        choices=["None", "Codeformer", "DMDNet", "GFPGAN", "GPEN", "Restoreformer++"],
        default="None",
        help="Post-processing face enhancer.",
    )
    parser.add_argument(
        "--face-detection",
        choices=["First found", "All input faces", "All female", "All male", "All faces"],
        default="First found",
        help="Which faces to swap in the target.",
    )
    parser.add_argument(
        "--face-distance",
        type=float,
        default=0.65,
        help="Max face similarity threshold (lower is more similar).",
    )
    parser.add_argument(
        "--blend-ratio",
        type=float,
        default=0.65,
        help="Blend ratio between original and enhanced face (enhancer only).",
    )
    parser.add_argument(
        "--swap-steps", type=int, default=1, help="Number of times to apply the swap process."
    )
    parser.add_argument(
        "--upscale",
        choices=["128px", "256px", "512px"],
        default="128px",
        help="Resolution for the subsample upscale process.",
    )
    parser.add_argument(
        "--provider",
        choices=["cpu", "cuda", "dml"],
        default=None,
        help="Override the execution provider from config.yaml.",
    )
    return parser.parse_args()


def get_processing_plugins(masking_engine):
    """Gets the dictionary of processors to use."""
    processors = {"faceswap": {}}
    if roop.globals.selected_enhancer and roop.globals.selected_enhancer != 'None':
        enhancer_map = {
            'GFPGAN': 'gfpgan',
            'Codeformer': 'codeformer',
            'DMDNet': 'dmdnet',
            'GPEN': 'gpen',
            'Restoreformer++': 'restoreformer++'
        }
        if roop.globals.selected_enhancer in enhancer_map:
            processors[enhancer_map[roop.globals.selected_enhancer]] = {}
    return processors


def translate_swap_mode(dropdown_text):
    """Translates user-friendly swap mode text to internal keys."""
    return {
        "First found": "first",
        "All input faces": "all_input",
        "All female": "all_female",
        "All male": "all_male",
        "All faces": "all",
    }.get(dropdown_text, "first")


def process_single(args):
    """Processes a single source and target file based on provided arguments."""
    # 1. Load source face
    print("Analyzing source image...")
    source_faceset = FaceSet()
    source_data = extract_face_images(args.source, (False, 0))
    if not source_data:
        print(f"ERROR: No faces found in source image: {args.source}")
        return

    face = source_data[0][0]
    face.mask_offsets = (0, 0, 0, 0, 1, 20)  # Default mask offsets
    source_faceset.faces.append(face)
    roop.globals.INPUT_FACESETS = [source_faceset]
    roop.globals.TARGET_FACES = []

    # 2. Initialize Process Manager
    roop.globals.face_swap_mode = translate_swap_mode(args.face_detection)
    roop.globals.selected_enhancer = args.enhancer
    roop.globals.distance_threshold = args.face_distance
    roop.globals.blend_ratio = args.blend_ratio
    roop.globals.subsample_size = int(args.upscale[:3])

    options = ProcessOptions(
        swap_model=args.swap_model,
        processordefines=get_processing_plugins(None),
        face_distance=args.face_distance,
        blend_ratio=args.blend_ratio,
        swap_mode=roop.globals.face_swap_mode,
        selected_index=0,
        masking_text=None,
        imagemask=None,
        num_steps=args.swap_steps,
        subsample_size=roop.globals.subsample_size,
        show_face_area=False,
        restore_original_mouth=False # Simplified for CLI, can be added as arg
    )

    process_mgr = ProcessMgr(progress=None)  # No Gradio progress bar for CLI
    process_mgr.initialize(roop.globals.INPUT_FACESETS, roop.globals.TARGET_FACES, options)

    # 3. Process Target
    if util.has_image_extension(args.target):
        print(f"Processing target image: {args.target}")
        target_frame = get_image_frame(args.target)
        if target_frame is None:
            print(f"ERROR: Failed to read target image: {args.target}")
            return

        result_frame = process_mgr.process_frame(target_frame)

        if result_frame is not None:
            print(f"Saving output to: {args.output}")
            cv2.imwrite(args.output, result_frame)
        else:
            print("ERROR: Processing failed, no result frame was produced.")

    elif util.is_video(args.target):
        print(f"Processing target video: {args.target}")
        total_frames = get_video_frame_total(args.target)
        fps = util.detect_fps(args.target)
        temp_output = args.output + ".tmp." + roop.globals.CFG.output_video_format

        process_mgr.run_batch_inmem(
            "File", args.target, temp_output, 0, total_frames, fps, roop.globals.CFG.max_threads
        )

        if os.path.exists(temp_output):
            print("Restoring audio...")
            restore_audio(temp_output, args.target, 0, total_frames, args.output)
            os.remove(temp_output)
            print(f"Video saved to {args.output}")
        else:
            print("ERROR: Video processing failed.")
    else:
        print(f"ERROR: Unsupported target file type: {args.target}")

    process_mgr.release_resources()
    print("Processing finished.")


def decode_execution_providers(providers):
    """Decodes provider names for ONNX Runtime."""
    available_providers = onnxruntime.get_available_providers()
    decoded_providers = []
    for provider in providers:
        if provider.lower() == 'cuda':
            if 'CUDAExecutionProvider' in available_providers:
                decoded_providers.append('CUDAExecutionProvider')
        elif provider.lower() == 'dml':
            if 'DmlExecutionProvider' in available_providers:
                decoded_providers.append('DmlExecutionProvider')
        else: # Default to CPU
            if 'CPUExecutionProvider' in available_providers:
                decoded_providers.append('CPUExecutionProvider')
    return decoded_providers or ['CPUExecutionProvider']

def run_cli():
    """Main entry point for the Command-Line Interface."""
    # Initialize settings from config file first
    roop.globals.CFG = Settings('config.yaml')
    
    args = parse_args()

    # Override provider from args if specified
    provider = args.provider if args.provider else roop.globals.CFG.provider
    if provider == "cuda" and not util.has_cuda_device():
        print("CUDA provider selected, but no CUDA device found. Falling back to CPU.")
        provider = "cpu"

    roop.globals.execution_providers = decode_execution_providers([provider])
    print(f"Using Execution Provider: {roop.globals.execution_providers[0]}")

    # Set output directory and ensure it exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Start processing
    process_single(args)

# This is the old main entry point, kept for reference but not used by the CLI.
def run():
    pass
