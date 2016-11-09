/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package contador_de_moedas;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.swing.JFileChooser;
import javax.swing.filechooser.FileNameExtensionFilter;

/**
 *
 * @author Cid
 */
public class Aquisicao  {
 
     String[] arquivos;

     public File openImagem(String Dir, boolean lista) {
        JFileChooser fileChooser = new JFileChooser();
        BufferedImage buff = null;
        String caminho = null;
        try {
            caminho = new File(".").getCanonicalPath() + Dir;
        } catch (IOException ex) {
            Logger.getLogger(Aquisicao.class.getName()).log(Level.SEVERE, null, ex);
        }
        fileChooser.setCurrentDirectory(new File(caminho));
        fileChooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
        fileChooser.addChoosableFileFilter(new FileNameExtensionFilter("Images", "jpg", "png", "gif", "bmp"));
        fileChooser.setAcceptAllFileFilterUsed(true);
        fileChooser.setMultiSelectionEnabled(true);
        int result = fileChooser.showOpenDialog(null);
        if (result == JFileChooser.APPROVE_OPTION) {
            if (lista) {
                File[] selectedFile = fileChooser.getSelectedFiles();
                arquivos = new String[selectedFile.length];
                for (int i = 0; i < selectedFile.length; i++) {
                    selectedFile[i].setReadable(Boolean.TRUE);
                    arquivos[i] = selectedFile[i].getAbsolutePath();
                }
            } else {
                File selectedFile = fileChooser.getSelectedFile();
                return selectedFile;
            }

        }
        return null;
    }
                                         
}
