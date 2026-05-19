import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def add_scale_bar_to_video_frame(
    video_path,
    scale_bar_length_um=100,
    pixel_size_um=1000/600,
    bar_position=(50, 550),
    bar_color=(255, 255, 255),
    bar_thickness=2,
    label_offset=(0, -10),
    font_scale=0.6,
    font_thickness=2,
    annotation_text=None,
    annotation_position=(350, 590),
    frame_index=0,
    display=True,
    figsize=(6, 6)
):
    """
    Add a scale bar to a video frame and optionally display it.
    
    Parameters:
    -----------
    video_path : str
        Path to the video file
    scale_bar_length_um : float
        Length of scale bar in micrometers
    pixel_size_um : float
        Size of one pixel in micrometers (default: 1000/600 for 1mm FOV over 600px)
    bar_position : tuple
        (x, y) position for the start of the scale bar
    bar_color : tuple
        BGR color for the scale bar and text
    bar_thickness : int
        Thickness of the scale bar line
    label_offset : tuple
        (x, y) offset for label position relative to bar start
    font_scale : float
        Font size scale
    font_thickness : int
        Font thickness
    annotation_text : str or None
        Additional annotation text (e.g., brain region)
    annotation_position : tuple
        (x, y) position for annotation text
    frame_index : int
        Which frame to extract (default: 0 for first frame)
    display : bool
        Whether to display the frame using matplotlib
    figsize : tuple
        Figure size for matplotlib display
        
    Returns:
    --------
    frame_with_bar : numpy.ndarray
        The frame with scale bar added
    """
    # Open the video
    cap = cv2.VideoCapture(video_path)
    
    # Navigate to the desired frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    
    # Read the frame
    ret, frame = cap.read()
    
    if not ret:
        cap.release()
        raise ValueError(f"Could not read frame {frame_index} from video")
    
    # Calculate scale bar length in pixels
    scale_bar_length_px = int(scale_bar_length_um / pixel_size_um)
    
    # Define scale bar endpoints
    start_point = bar_position
    end_point = (bar_position[0] + scale_bar_length_px, bar_position[1])
    
    # Draw the scale bar
    frame_with_bar = cv2.line(frame.copy(), start_point, end_point, bar_color, bar_thickness)
    
    # Add scale bar label
    label = f'{scale_bar_length_um} um'
    label_position = (bar_position[0] + label_offset[0], bar_position[1] + label_offset[1])
    cv2.putText(frame_with_bar, label, label_position, 
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, bar_color, font_thickness)
    
    # Add annotation text if provided
    if annotation_text:
        cv2.putText(frame_with_bar, annotation_text, annotation_position,
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, bar_color, font_thickness)
    
    # Display the frame if requested
    if display:
        plt.figure(figsize=figsize)
        plt.axis('off')
        plt.imshow(cv2.cvtColor(frame_with_bar, cv2.COLOR_BGR2RGB))
        plt.show()
    
    # Release the video capture object
    cap.release()
    
    return frame_with_bar


# Example usage:

# video_path = "/data/big_rim/For tech paper/20250626PMC2_L23 (on Nov 14, 2025)/0.avi"

# frame_with_bar = add_scale_bar_to_video_frame(
#     video_path=video_path,
#     scale_bar_length_um=100,
#     annotation_text='Primary Visual Cortex'
# )





def add_scale_bar_to_video_frame_save(
    video_path,
    scale_bar_length_um=100,
    pixel_size_um=1000/600,
    bar_position=(50, 550),
    bar_color=(255, 255, 255),
    bar_thickness=2,
    label_offset=(0, -10),
    font_scale=0.6,
    font_thickness=2,
    annotation_text=None,
    annotation_position=(350, 590),
    frame_index=0,
    display=True,
    figsize=(6, 6),
    save_path=None,
    save_format='png',
    dpi=300
):
    """
    Add a scale bar to a video frame and optionally display/save it.
    
    Parameters:
    -----------
    video_path : str
        Path to the video file
    scale_bar_length_um : float
        Length of scale bar in micrometers
    pixel_size_um : float
        Size of one pixel in micrometers (default: 1000/600 for 1mm FOV over 600px)
    bar_position : tuple
        (x, y) position for the start of the scale bar
    bar_color : tuple
        BGR color for the scale bar and text
    bar_thickness : int
        Thickness of the scale bar line
    label_offset : tuple
        (x, y) offset for label position relative to bar start
    font_scale : float
        Font size scale
    font_thickness : int
        Font thickness
    annotation_text : str or None
        Additional annotation text (e.g., brain region)
    annotation_position : tuple
        (x, y) position for annotation text
    frame_index : int
        Which frame to extract (default: 0 for first frame)
    display : bool
        Whether to display the frame using matplotlib
    figsize : tuple
        Figure size for matplotlib display
    save_path : str or None
        Path to save the output image. If None, image is not saved.
    save_format : str
        Format to save the image. Options: 'png', 'jpg', 'tiff', 'pdf', 'svg', 'eps'
        Note: 'pdf', 'svg', 'eps' are vector formats
    dpi : int
        Resolution for saved image (relevant for raster formats)
        
    Returns:
    --------
    frame_with_bar : numpy.ndarray
        The frame with scale bar added
    """
    # Open the video
    cap = cv2.VideoCapture(video_path)
    
    # Navigate to the desired frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    
    # Read the frame
    ret, frame = cap.read()
    
    if not ret:
        cap.release()
        raise ValueError(f"Could not read frame {frame_index} from video")
    
    # Calculate scale bar length in pixels
    scale_bar_length_px = int(scale_bar_length_um / pixel_size_um)
    
    # Define scale bar endpoints
    start_point = bar_position
    end_point = (bar_position[0] + scale_bar_length_px, bar_position[1])
    
    # Draw the scale bar
    frame_with_bar = cv2.line(frame.copy(), start_point, end_point, bar_color, bar_thickness)
    
    # Add scale bar label
    label = f'{scale_bar_length_um} um'
    label_position = (bar_position[0] + label_offset[0], bar_position[1] + label_offset[1])
    cv2.putText(frame_with_bar, label, label_position, 
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, bar_color, font_thickness)
    
    # Add annotation text if provided
    if annotation_text:
        cv2.putText(frame_with_bar, annotation_text, annotation_position,
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, bar_color, font_thickness)
    
    # Convert BGR to RGB for matplotlib
    frame_rgb = cv2.cvtColor(frame_with_bar, cv2.COLOR_BGR2RGB)
    
    # Display and/or save
    if display or save_path:
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis('off')
        ax.imshow(frame_rgb)
        
        if save_path:
            # Add file extension if not provided
            if not any(save_path.endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.tiff', '.pdf', '.svg', '.eps']):
                save_path = f"{save_path}.{save_format}"
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
            
            # Save with appropriate settings
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
            print(f"Saved to: {save_path}")
        
        if display:
            plt.show()
        else:
            plt.close(fig)
    
    # Release the video capture object
    cap.release()
    
    return frame_with_bar


# Example usage:

    # video_path = "/data/big_rim/For tech paper/20250626PMC2_L23 (on Nov 14, 2025)/0.avi"
    
    # # Save as PNG
    # frame_with_bar = add_scale_bar_to_video_frame(
    #     video_path=video_path,
    #     scale_bar_length_um=100,
    #     annotation_text='Primary Visual Cortex',
    #     save_path='output_frame.png',
    #     save_format='png',
    #     dpi=300
    # )
    
    # # Save as vector (PDF)
    # frame_with_bar = add_scale_bar_to_video_frame(
    #     video_path=video_path,
    #     scale_bar_length_um=100,
    #     annotation_text='Primary Visual Cortex',
    #     save_path='output_frame.pdf',
    #     save_format='pdf',
    #     display=False
    # )
    
    # # Save as SVG (vector)
    # frame_with_bar = add_scale_bar_to_video_frame(
    #     video_path=video_path,
    #     scale_bar_length_um=100,
    #     annotation_text='Primary Visual Cortex',
    #     save_path='output_frame.svg',
    #     save_format='svg',
    #     display=False
    # )