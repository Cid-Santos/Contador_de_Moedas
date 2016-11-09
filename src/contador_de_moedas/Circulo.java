/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package contador_de_moedas;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.imageio.ImageIO;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import static org.opencv.imgproc.Imgproc.CV_COMP_CORREL;

/**
 *
 * @author Cid
 */
public class Circulo {

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    private final String imgS;
    private final File imgFB, imgFA;
    private Mat imageB, imageA;
    private Mat output;

    public List<String> resultados = new ArrayList<>();
    public int[] rowsCols = new int[2];
    private final Mat circles = new Mat();

    private double param1;
    private double param2;
    private int minRadius;
    private int maxRadius;
    private boolean criaBase = false;
    private int somatorio = 0;

    /**
     * @param param1 the param1 to set
     */
    public void setParam1(double param1) {
        this.param1 = param1;
    }

    /**
     * @param param2 the param2 to set
     */
    public void setParam2(double param2) {
        this.param2 = param2;
    }

    /**
     * @param minRadius the minRadius to set
     */
    public void setMinRadius(int minRadius) {
        this.minRadius = minRadius;
    }

    /**
     * @param maxRadius the maxRadius to set
     */
    public void setMaxRadius(int maxRadius) {
        this.maxRadius = maxRadius;
    }

    /**
     * @param criaBase the criaBase to set
     */
    public void setCriaBase(boolean criaBase) {
        this.criaBase = criaBase;
    }

    public Circulo(File imgFB, File imgFA) {
        this.imgFB = imgFB;
        this.imgFA = imgFA;
        this.imgS = this.imgFB.getAbsolutePath();
        LoadImagA();
        LoadImagB();
        IniciaImag();
    }

    private void IniciaImag() {
        this.output = new Mat(imageB.height(), imageB.width(), CvType.CV_8UC3);
    }

    private void LoadImagB() {
        try {
            BufferedImage image = ImageIO.read(imgFB);
            byte[] data = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
            imageB = new Mat(image.getHeight(), image.getWidth(), CvType.CV_8UC3);
            imageB.put(0, 0, data);
        } catch (IOException ex) {
            Logger.getLogger(Circulo.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private void LoadImagA() {
        try {
            BufferedImage image = ImageIO.read(imgFA);
            byte[] data = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
            imageA = new Mat(image.getHeight(), image.getWidth(), CvType.CV_8UC3);
            imageA.put(0, 0, data);
        } catch (IOException ex) {
            Logger.getLogger(Circulo.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    Mat[] vetMatFile;
    Mat[] vetMat = new Mat[100];

    private Mat doCirciloHough(Mat imgIn) {
        int valor = 0;
        baseImageList();
        Reconhecimento _reco;
        Imgproc.cvtColor(imgIn, imgIn, Imgproc.COLOR_BGR2GRAY);
        /*Imgproc.erode(imgIn, imgIn, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(2, 2)));
         Imgproc.dilate(imgIn, imgIn, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(2, 2)));*/

        double min_dist_Centro = imgIn.rows() / 8.0;
        /*
         Imgproc.HoughCircles(imageB, circles, method, dp, minDist, param1, param2, minRadius, maxRadius)
         imageB: Imagem de entrada (tons de cinza)
         circles: um vector que armazena conjuntos de 3 valores: Xc, Yc, r para cada círculo detectado.
         method: CV_HOUGH_GRADIENT definir o método de detecção. Atualmente, esse é o único disponível em OpenCV
         dp: (1) A razão inversa da resolução
         min_dist: (imgIn.rows / 8 ) A distância mínima entre centros detectados
         param1: (200 limite superior para o detector de bordas Canny interna) limite superior para o detector de bordas Canny interna
         param2: (100* Limiar para detecção de centro)Limiar para detecção de centro.
         minRadius: (0) raio minimo a ser detectado. Se for desconhecida, colocar zero como padrao.
         maxRadius: (maxRadius) raio maximo a ser detectado. Se desconhecida, colocar zero como padrao   
         */
        Imgproc.HoughCircles(imgIn, circles, Imgproc.CV_HOUGH_GRADIENT, 1, min_dist_Centro, param1, param2, minRadius, maxRadius);

        rowsCols[0] = circles.rows();
        rowsCols[1] = circles.cols();

        for (int i = 0; i < circles.cols(); i++) {
            _reco = new Reconhecimento();
            double data[] = circles.get(0, i);
            Point pt = new Point(Math.round(data[0]), Math.round(data[1]));
            int raio = (int) Math.round(data[2]);

            if (data[2] > 20.0 && data[2] < 28.0) {
                valor = 10;
                setSomatorio(getSomatorio() + valor);
            } else if (data[2] > 27.0 && data[2] < 32.0) {
                valor = 5;
                setSomatorio(getSomatorio() + valor);
            } else if (data[2] > 32.0 && data[2] < 33.0) {
                valor = 50;
                setSomatorio(getSomatorio() + valor);
            } else if (data[2] > 33.5 && data[2] < 36.0) {
                valor = 25;
                setSomatorio(getSomatorio() + valor);
            } else if (data[2] > 35.0 && data[2] < 40.0) {
                valor = 100;
                setSomatorio(getSomatorio() + valor);
            }
            resultados.add(" r:" + (int) raio + "   (X:" + (int) pt.x + "-Y:" + (int) pt.y + ") Val: " + valor);
            vetMat[i] = _reco.CriaMascara(pt, data[2], imgIn);
            output = _reco.CalculaHistograma(vetMat[i], imageA);
            
            
            /*  System.out.println("histogram\n"+output.dump());
             for (int j = 0; j < vetMatFile.length - 1; j++) {
             Mat ab = vetMatFile[j];
             System.out.println("histogram\n"+vetMatFile[j].dump());
             double distance = Imgproc.compareHist(output, vetMatFile[j], Imgproc.CV_COMP_CORREL);

             System.out.print("ok");
             }*/

            if (criaBase) {
                FileWriter arq = null;
                try {
                    String nome = "X" + (int) pt.x + "Y" + (int) pt.y + "R" + (int) raio;
                    // Imgcodecs.imwrite("baseConhecimento/" + nome + "M.png", vetMat[i]);
                    arq = new FileWriter("baseConhecimento/" + nome + ".txt");
                    PrintWriter gravarArq = new PrintWriter(arq);
                    gravarArq.print(output.dump());
                    arq.close();
                    // Imgcodecs.imwrite("baseConhecimento/" + nome + ".yml", output);
                } catch (IOException ex) {
                    Logger.getLogger(Circulo.class.getName()).log(Level.SEVERE, null, ex);
                } finally {
                    try {
                        arq.close();
                    } catch (IOException ex) {
                        Logger.getLogger(Circulo.class.getName()).log(Level.SEVERE, null, ex);
                    }
                }
            }

            Imgproc.circle(imageA, pt, raio, new Scalar(0, 0, 255), 2);
            Imgproc.circle(imageA, pt, 2, new Scalar(0, 255, 0), 1);
        }
        return imageA;
    }

    private void baseImageList() {

        File dir = new File("baseConhecimento/");
        if (dir.isDirectory()) {
            String[] arqs = dir.list();
            vetMatFile = new Mat[arqs.length];
            for (int i = 0; i < arqs.length; i++) {
                String nome = "baseConhecimento/" + arqs[i];
                vetMatFile[i] = Imgcodecs.imread(nome, CvType.CV_8UC3);
            }
        }
    }

    public String _CirciloHough() {
        String result = null;
        output = doCirciloHough(imageB);
        Collections.sort(resultados);
        String nome = (imgS.substring(imgS.lastIndexOf("\\") + 1, imgS.length()));
        if (nome.contains("HoughCircles_")) {
            result = "OutputImg/" + (imgS.substring(imgS.lastIndexOf("\\") + 1, imgS.length()));
        } else {
            result = "OutputImg/HoughCircles_" + (imgS.substring(imgS.lastIndexOf("\\") + 1, imgS.length()));
        }
        Imgcodecs.imwrite(result, output);
        return result;
    }

    /**
     * @return the somatorio
     */
    public int getSomatorio() {
        return somatorio;
    }

    /**
     * @param somatorio the somatorio to set
     */
    public void setSomatorio(int somatorio) {
        this.somatorio = somatorio;
    }

}
