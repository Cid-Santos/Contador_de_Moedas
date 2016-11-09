/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package contador_de_moedas;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.imageio.ImageIO;
import javax.swing.JSlider;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

/**
 *
 * @author Cid
 */
public class Segmentacao {

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    // canny threshold value
    private final JSlider threshold;

    private final String imgS;
    private final File imgF;
    private Mat image;
    private Mat output;

    public Segmentacao(File imgF, JSlider slider) {
        this.imgF = imgF;
        this.imgS = imgF.getAbsolutePath();
        this.threshold = slider;
        LoadImagA();
        IniciaImag();
    }

    private void IniciaImag() {
        this.output = new Mat(image.width(), image.height(), CvType.CV_8UC3);
    }

    private void LoadImagA() {
        try {
            BufferedImage imageA = ImageIO.read(imgF);
            byte[] data = ((DataBufferByte) imageA.getRaster().getDataBuffer()).getData();
            image = new Mat(imageA.getHeight(), imageA.getWidth(), CvType.CV_8UC3);
            image.put(0, 0, data);
            image = Binarizar(image);
        } catch (IOException ex) {
            Logger.getLogger(Segmentacao.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private Mat Binarizar(Mat imgOriginal) {
        Mat Cinza = new Mat();
        Mat Binarizada = new Mat();
         // convert to grayscale
        Imgproc.cvtColor(imgOriginal, Cinza, Imgproc.COLOR_BGR2GRAY);
        //Binarizacao de imagem 
        Imgproc.adaptiveThreshold(Cinza, Binarizada, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY_INV, 15, 20);
        return (Binarizada);
    }

    /**
     * Apply Canny
     *
     * @param img the current frame
     * @return an image elaborated with Canny
     */
    private Mat doCanny(Mat img) {
        // inicia
        Mat grayImage = new Mat();
        Mat detectedEdges = new Mat();

        // Reduzir o ruído com um kernel 3x3
        Imgproc.blur(grayImage, detectedEdges, new Size(3, 3));

        //Detector canny, com relação de menor: limite de superior de 3: 1
        Imgproc.Canny(detectedEdges, detectedEdges, this.threshold.getValue(), this.threshold.getValue() * 3);

        //Usando Canny e exibindo o resultadosaída como uma máscara. 
        Mat dest = new Mat();
        img.copyTo(dest, detectedEdges);

        return dest;
    }

    public String _canny() {
        output = doCanny(image);
        String result = "OutputImg/" + imgS.substring(imgS.lastIndexOf("\\") + 1, imgS.length());
        Imgcodecs.imwrite(result, output);
        return result;
    }
}
