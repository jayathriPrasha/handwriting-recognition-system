# handwriting-recognition-system
Overcoming handwriting recognition challenges in an online environment is a multifaceted problem. Improve 75% accuracy of handwriting recognition for grammar check 

*How  this implement 
Creating a system diagram for this script involves illustrating the main components and their interactions. Here's a simplified system diagram for the image segmentation script:


                +---------------------+
                | Read Grayscale Image|
                |    (cv2.imread)      |
                +----------+----------+
                           |
                           v
                +---------------------+
                |   Set Threshold     |
                |   (threshold = 250)  |
                +----------+----------+
                           |
                           v
                +---------------------+
                | Initialize Variables|
                |   (segment_start,    |
                |    segment_end)      |
                +----------+----------+
                           |
                           v
                +---------------------+
                | Create Segments List|
                |    (segments = [])   |
                +----------+----------+
                           |
                           v
                +---------------------+
                |   Segmentation Loop |
                |   (for each row)     |
                +----------+----------+
                           |
                           v
                +---------------------+
                |   Save Segments     |
                |   (cv2.imwrite)      |
                +----------+----------+
                           |
                           v
                +---------------------+
                | Print Number of     |
                |    Segments Found   |
                +---------------------+






*image input in to system

1.Read Grayscale Image:

2.Reads an image in grayscale from the specified file path (thresholded.png).
Set Threshold:
![preprocessed_image](https://github.com/jayathriPrasha/handwriting-recognition-system/assets/141565380/58f8295a-0f43-458d-abb9-d1b0dcdbf243)
3.Sets a threshold value (250) to determine whether a row's mean intensity is considered part of a segment.
Initialize Variables:

4.Initializes variables to keep track of the start and end of a segment.
Create a List to Store Segments:

5.Initializes an empty list (segments) to store the segmented portions of the image.
Segmentation Loop:

6.Iterates through each row of the image.
Calculates the mean intensity of each row.
If the row's mean intensity is below the threshold and it's the start of a new segment, it records the start row.
If the row's mean intensity is above the threshold and it was previously in a segment, it records the end row and extracts the segment.
The extracted segment is stored in the segments list.
Save Segments to Files:

7.Iterates through the list of segments and saves each segment as a separate PNG file in the "data" directory with filenames like segment_0.png, segment_1.png, etc.
Print the Number of Segments:

8Prints the number of segments found.









*output show image how it work
1.Check Request Method:

2.If the request method is POST, it means a form was submitted.
Process Image Data:

3.Retrieve image data from the JSON in the request.
Save the image to the "uploads" directory.
Preprocess Image:

4.Call the preprocess function.
OCR on Image Segments:

5.List files in the "segments" directory.
Perform OCR on each segment using the inference_ocr function.
Concatenate the OCR results into output_text.
Delete Temporary Files:

6.List files in the "segments" directory.
Delete each file.
Correct Text:

7.Use the corrector function to improve the OCR output.
Return Results:

8.Return a JSON response containing both the original (output_text) and corrected text.





#how it work vedio 
