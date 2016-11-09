/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package contador_de_moedas;

import java.util.ArrayList;
import java.util.List;
import java.util.Vector;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import static org.opencv.core.CvType.CV_8U;
import static org.opencv.core.CvType.CV_8UC1;
import org.opencv.core.Mat;
import static org.opencv.core.Mat.zeros;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

/**
 *
 * @author Cid
 */
public class Reconhecimento {

    public Mat CriaMascara(Point pt, double raio, Mat img_Original) {
        // Cria uma img toda preta com as mesmas dimensões da original
        Mat mask = zeros(img_Original.rows(), img_Original.cols(), CV_8U);
        // Cria um circulo branco preenchido
        Imgproc.circle(mask, pt, (int) raio, new Scalar(255, 255, 255), -1, 8, 0);
        return (mask);
    }

    public Mat CalculaHistograma(Mat mask, Mat img_Original) {
        // Converte para um tipo de imagem com canais separados
        List<Mat> imagesList = new ArrayList<>();
        imagesList.add(img_Original);
        // Parametros do calcHist
        Mat histograma = new Mat();
        int h_bins = 30;
        int s_bins = 32;
        MatOfInt mHistSize = new MatOfInt(h_bins, s_bins);
        MatOfFloat mRanges = new MatOfFloat(0, 179, 0, 255);
        MatOfInt mChannels = new MatOfInt(0, 1);
        // Calcula o histograma
        Imgproc.calcHist(imagesList, mChannels, mask, histograma, mHistSize, mRanges, false);
        // Normalização necessária para ajustar valores
        Core.normalize(histograma, histograma, 0, 255, Core.NORM_MINMAX, -1, new Mat());

        return (histograma);
    }

}
