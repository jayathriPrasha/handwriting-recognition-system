# handwriting-recognition-system
Improve 75% accuracy of handwriting recognition for gammer check 

how  this implement 
Creating a system diagram for this script involves illustrating the main components and their interactions. Here's a simplified system diagram for the image segmentation script:

```
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
```

This diagram outlines the flow of the script, from reading the grayscale image to printing the number of segments found. Each box represents a step or component, and the arrows indicate the flow of data or control between them.

Note: This is a high-level overview, and you can further detail each component or add more information based on the specific functionalities and dependencies in your system.
![preprocessed_image](https://github.com/jayathriPrasha/handwriting-recognition-system/assets/141565380/58f8295a-0f43-458d-abb9-d1b0dcdbf243)
