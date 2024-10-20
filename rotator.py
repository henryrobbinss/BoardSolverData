import json
import cv2
import numpy as np

# Function to draw bounding boxes on an image
def draw_bounding_boxes(image, annotations):
    for annotation in annotations:
        x, y, width, height = annotation['coordinates'].values()
        label = annotation['label']
        # Draw the rectangle (bounding box) on the image
        top_left = (int(x-(float(width)/2.0)), int(y-(float(height)/2.0)))
        bottom_right = (int(x + (float(width)/2.0)), int(y + (float(height)/2.0)))
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
        # Put the label above the box
        cv2.putText(image, label, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

# Function to rotate an image and its bounding boxes
def rotate_image_and_bboxes(image_path, annotations, angle):
    # Load the image
    image = cv2.imread(image_path)
    
    # Get image dimensions
    (h, w) = image.shape[:2]
    
    # Compute the center of the image
    center = (w // 2, h // 2)
    
    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, M, (w, h))
    
    # Rotate the bounding boxes
    for annotation in annotations:
        x, y, width, height = annotation['coordinates'].values()
        box = np.array([
            [x, y],
            [x + width, y],
            [x + width, y + height],
            [x, y + height]
        ])
        ones = np.ones(shape=(len(box), 1))
        points_ones = np.hstack([box, ones])
        rotated_box = M.dot(points_ones.T).T
        
        # Update the bounding box in the annotation
        annotation['coordinates'] = {
            'x': rotated_box[:, 0].min(),
            'y': rotated_box[:, 1].min(),
            'width': rotated_box[:, 0].max() - rotated_box[:, 0].min(),
            'height': rotated_box[:, 1].max() - rotated_box[:, 1].min()
        }
    
    return rotated_image, annotations

# Load your JSON data
with open('_annotations.createml.json') as f:
    data = json.load(f)

# Iterate through each image in the dataset
for item in data:
    image_path = item['image']
    annotations = item['annotations']
    
    # Show the original image with bounding boxes
    image = cv2.imread(image_path)
    image_with_boxes = draw_bounding_boxes(image.copy(), annotations)
    
    cv2.imshow("Original Image with Bounding Boxes", image_with_boxes)
    cv2.waitKey(0)  # Wait for key press to move to the next image
    cv2.destroyAllWindows()
    
    # Manually input the rotation angle
    angle = float(input(f"Enter rotation angle for {image_path} (e.g., 90, -90, 180): "))
    
    # Rotate the image and bounding boxes if necessary
    if angle != 0:
        rotated_image, rotated_annotations = rotate_image_and_bboxes(image_path, annotations, angle)
        
        # Show the rotated image with updated bounding boxes
        rotated_image_with_boxes = draw_bounding_boxes(rotated_image.copy(), rotated_annotations)
        cv2.imshow("Rotated Image with Bounding Boxes", rotated_image_with_boxes)
        cv2.waitKey(0)  # Wait for key press to move to the next image
        cv2.destroyAllWindows()
        
        output_image_path = f"rotated_{image_path}"
        cv2.imwrite(output_image_path, rotated_image)
        
        # Update the annotations with the rotated bounding boxes
        item['image'] = output_image_path
        item['annotations'] = rotated_annotations

# Save the updated JSON with the rotated annotations
with open('updated_annotations.json', 'w') as f:
    json.dump(data, f, indent=4)
