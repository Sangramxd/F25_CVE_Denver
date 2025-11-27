import cv2 as cv

def blockage_pipeline(image_path):
    
    # load the image
    img = cv.imread(image_path)
    
    # Cut out border
    img = img[20:-20, 20:-20]
    
    # convert to grayscale image
    grayscale_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # brighten the image
    grayscale_image = cv.convertScaleAbs(grayscale_image, alpha=1.25, beta=15)
    
    # show the grayscale image
    cv.imshow("Grayscale Image", grayscale_image)
    cv.waitKey(0)
    
    # pseudocolor the image
    pseudocolored_image = cv.applyColorMap(grayscale_image, cv.COLORMAP_JET)
    
    # show the pseudocolored image
    cv.imshow("Pseudocolored Image", pseudocolored_image)
    cv.waitKey(0)
    
    # detect dark blue regions (potential blockages)
    lower_blue = (100, 0, 0)
    upper_blue = (255, 100, 100)
    mask = cv.inRange(pseudocolored_image, lower_blue, upper_blue)
    
    # find contours of the masked regions
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(image = img, contours = contours, contourIdx = -1, color = (0, 255, 0), thickness = 2)
    
    # redraw found contours on original image
    cv.drawContours(image = img, contours = contours, contourIdx = -1, color = (0, 255, 0), thickness = 2)
    
    # save the final image with detected blockages
    output_path = image_path.replace('.png', '_blockages_detected.png')
    cv.imwrite(output_path, img)
    
    # show the final image with detected blockages
    cv.imshow("Detected Blockages", img)
    cv.waitKey(0)
    

# Example usage
if __name__ == "__main__":
    blockage_pipeline("Angiogram_1.png")
    blockage_pipeline("Angiogram_2.png")
    blockage_pipeline("Angiogram_3.png")