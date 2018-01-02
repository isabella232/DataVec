/*-
 *  * Copyright 2016 Skymind, Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 */

package org.datavec.image.loader;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.linalg.util.NDArrayUtil;

import java.io.*;
import java.nio.ByteBuffer;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.image.PixelFormat;
import javafx.scene.image.PixelReader;
import javafx.scene.image.PixelWriter;
import javafx.scene.image.WritableImage;
import javafx.scene.image.WritablePixelFormat;
import javafx.scene.paint.Color;

/**
 * Image loader for taking images
 * and converting them to matrices
 * @author Adam Gibson
 *
 */
public class ImageLoader extends BaseImageLoader {

    public ImageLoader() {
        super();
    }

    /**
     * Instantiate an image with the given
     * height and width
     * @param height the height to load*
     * @param width  the width to load
    
     */
    public ImageLoader(int height, int width) {
        super();
        this.height = height;
        this.width = width;
    }


    /**
     * Instantiate an image with the given
     * height and width
     * @param height the height to load
     * @param width  the width to load
     * @param channels the number of channels for the image*
     */
    public ImageLoader(int height, int width, int channels) {
        super();
        this.height = height;
        this.width = width;
        this.channels = channels;
    }

    /**
     * Instantiate an image with the given
     * height and width
     * @param height the height to load
     * @param width  the width to load
     * @param channels the number of channels for the image*
     * @param centerCropIfNeeded to crop before rescaling and converting
     */
    public ImageLoader(int height, int width, int channels, boolean centerCropIfNeeded) {
        this(height, width, channels);
        this.centerCropIfNeeded = centerCropIfNeeded;
    }

    /**
     * Convert a file to a row vector
     *
     * @param f the image to convert
     * @return the flattened image
     * @throws IOException
     */
    @Override
    public INDArray asRowVector(File f) throws IOException {
        return asRowVector(new Image(f.toURI().toString()));
    }

    @Override
    public INDArray asRowVector(InputStream inputStream) throws IOException {
        return asRowVector(new Image(inputStream));
    }

    /**
     * Convert an image in to a row vector
     * @param image the image to convert
     * @return the row vector based on a rastered
     * representation of the image
     */
    public INDArray asRowVector(Image image) {
        if (centerCropIfNeeded) {
            image = centerCropIfNeeded(image);
        }
        image = scalingIfNeed(image, true);
        if (channels == 3) {
            return toINDArrayBGR(image).ravel();
        }
        int[][] ret = toIntArrayArray(image);
        return NDArrayUtil.toNDArray(ArrayUtil.flatten(ret));
    }

    /**
     * Changes the input stream in to an
     * bgr based raveled(flattened) vector
     * @param file the input stream to convert
     * @return  the raveled bgr values for this input stream
     */
    public INDArray toRaveledTensor(File file) {
        try {
            INDArray ret = toRaveledTensor(new FileInputStream(file));
            return ret.ravel();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Changes the input stream in to an
     * bgr based raveled(flattened) vector
     * @param is the input stream to convert
     * @return  the raveled bgr values for this input stream
     */
    public INDArray toRaveledTensor(InputStream is) {
        return toBgr(is).ravel();
    }

    /**
     * Convert an image in to a raveled tensor of
     * the bgr values of the image
     * @param image the image to parse
     * @return the raveled tensor of bgr values
     */
    public INDArray toRaveledTensor(Image image) {
        try {
            image = scalingIfNeed(image, false);
            return toINDArrayBGR(image).ravel();
        } catch (Exception e) {
            throw new RuntimeException("Unable to load image", e);
        }
    }

    /**
     * Convert an input stream to an bgr spectrum image
     *
     * @param file the file to convert
     * @return the input stream to convert
     */
    public INDArray toBgr(File file) {
        try {
            INDArray ret = toBgr(new FileInputStream(file));
            return ret;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Convert an input stream to an bgr spectrum image
     *
     * @param inputStream the input stream to convert
     * @return the input stream to convert
     */
    public INDArray toBgr(InputStream inputStream) {
        return toBgr(new Image(inputStream));
    }

    private org.datavec.image.data.Image toBgrImage(InputStream inputStream){
        Image image = new Image(inputStream);
        INDArray img = toBgr(image);
        int bands = getNumBands(image.getPixelReader());
        return new org.datavec.image.data.Image(img, bands, (int) image.getHeight(), (int) image.getWidth());
    }

    /**
     * Convert an Image to an bgr spectrum image
     *
     * @param image the Image to convert
     * @return the input stream to convert
     */
    public INDArray toBgr(Image image) {
        if (image == null)
            throw new IllegalStateException("Unable to load image");
        image = scalingIfNeed(image, false);
        return toINDArrayBGR(image);
    }

    /**
     * Convert an image file
     * in to a matrix
     * @param f the file to convert
     * @return a 2d matrix of a rastered version of the image
     * @throws IOException
     */
    @Override
    public INDArray asMatrix(File f) throws IOException {
        return NDArrayUtil.toNDArray(fromFile(f));
    }

    /**
     * Convert an input stream to a matrix
     * @param inputStream the input stream to convert
     * @return the input stream to convert
     */
    @Override
    public INDArray asMatrix(InputStream inputStream) {
        if (channels == 3) {
            return toBgr(inputStream);
        }
        Image image = new Image(inputStream);
        return asMatrix(image);
    }

    @Override
    public org.datavec.image.data.Image asImageMatrix(File f) throws IOException {
        try (BufferedInputStream bis = new BufferedInputStream(new FileInputStream(f))) {
            return asImageMatrix(bis);
        }
    }

    @Override
    public org.datavec.image.data.Image asImageMatrix(InputStream inputStream) throws IOException {
        if (channels == 3) {
            return toBgrImage(inputStream);
        }
        Image image = new Image(inputStream);
        INDArray asMatrix = asMatrix(image);
        return new org.datavec.image.data.Image(asMatrix, getNumBands(image.getPixelReader()), (int) image.getHeight(), (int) image.getWidth());
    }

    /**
     * Convert an Image to a matrix
     * @param image the Image to convert
     * @return the input stream to convert
     */
    public INDArray asMatrix(Image image) {
        if (channels == 3) {
            return toBgr(image);
        } else {
            image = scalingIfNeed(image, true);
            int w = (int) image.getWidth();
            int h = (int) image.getHeight();
            INDArray ret = Nd4j.create(h, w);

            PixelReader reader = image.getPixelReader();
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    ret.putScalar(new int[] {i, j}, reader.getArgb(j, i));
                }
            }
            return ret;
        }
    }

    /**
     * Slices up an image in to a mini batch.
     *
     * @param f the file to load from
     * @param numMiniBatches the number of images in a mini batch
     * @param numRowsPerSlice the number of rows for each image
     * @return a tensor representing one image as a mini batch
     */
    public INDArray asImageMiniBatches(File f, int numMiniBatches, int numRowsPerSlice) {
        try {
            INDArray d = asMatrix(f);
            return Nd4j.create(numMiniBatches, numRowsPerSlice, d.columns());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public int[] flattenedImageFromFile(File f) throws IOException {
        return ArrayUtil.flatten(fromFile(f));
    }

    /**
     * Load a rastered image from file
     * @param file the file to load
     * @return the rastered image
     * @throws IOException
     */
    public int[][] fromFile(File file) throws IOException {
        Image image = new Image(file.toURI().toString());
        image = scalingIfNeed(image, true);
        return toIntArrayArray(image);
    }

    /**
     * Load a rastered image from file
     * @param file the file to load
     * @return the rastered image
     * @throws IOException
     */
    public int[][][] fromFileMultipleChannels(File file) throws IOException {
        Image image = new Image(file.toURI().toString());
        image = scalingIfNeed(image, channels > 3);

        int w = (int) image.getWidth();
        int h = (int) image.getHeight();
        PixelReader reader = image.getPixelReader();
        int bands = getNumBands(reader);
        int[][][] ret = new int[channels][h][w];
        byte[] pixels = new byte[w * h * bands];
        reader.getPixels(0, 0, w, h, (WritablePixelFormat<ByteBuffer>) reader.getPixelFormat(), pixels, 0, w * bands);
        
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                for (int k = 0; k < channels; k++) {
                    if (k >= bands)
                        break;
                    ret[k][i][j] = pixels[channels * w * i + channels * j + k];
                }
            }
        }
        return ret;
    }

    /**
     * Convert a matrix in to a buffereed image
     * @param matrix the
     * @return {@link Image}
     */
    public static Image toImage(INDArray matrix) {
        WritableImage writableImage = new WritableImage(matrix.rows(), matrix.columns());
        PixelWriter pixelWriter = writableImage.getPixelWriter();
        for (int i = 0; i < matrix.rows(); i++) {
            for (int j = 0; j < matrix.columns(); j++) {
                pixelWriter.setArgb(j, i, (int) matrix.getDouble(j + i * matrix.columns()));
            }
        }
        return writableImage;
    }


    private static int[] rasterData(INDArray matrix) {
        int[] ret = new int[matrix.length()];
        for (int i = 0; i < ret.length; i++)
            ret[i] = (int) Math.round((double) matrix.getScalar(i).element());
        return ret;
    }

    /**
     * Convert the given image to an rgb image
     * @param arr the array to use
     * @param image the image to set
     */
    public void toImageRGB(INDArray arr, Image image) {
        if (arr.rank() < 3)
            throw new IllegalArgumentException("Arr must be 3d");

        image = scalingIfNeed(image, arr.size(-2), arr.size(-1), true);
        WritableImage writableImage = new WritableImage(image.getPixelReader(), width,  height);
        PixelWriter pixelWriter = writableImage.getPixelWriter();
        for (int i = 0; i < image.getWidth(); i++) {
            for (int j = 0; j < image.getHeight(); j++) {
                int r = arr.slice(0).getInt(i, j);
                int g = arr.slice(1).getInt(i, j);
                int b = arr.slice(2).getInt(i, j);
                int a = 1;
                int col = (a << 24) | (r << 16) | (g << 8) | b;
                pixelWriter.setArgb(i, j, col);
            }
        }
        image = writableImage;
    }

    protected int[][] toIntArrayArray(Image image) {
        int w = (int) image.getWidth(), h = (int) image.getHeight();
        int[][] ret = new int[h][w];
        PixelReader pixelReader = image.getPixelReader();
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                ret[i][j] = pixelReader.getArgb(j, i);
            }
        }
        return ret;
    }

    protected INDArray toINDArrayBGR(Image image) {
        int height = (int) image.getHeight();
        int width = (int) image.getWidth();
        final PixelReader pixelReader = image.getPixelReader();
        int bands = getNumBands(pixelReader);

        byte[] pixels = new byte[width * height * bands];
        final WritablePixelFormat<ByteBuffer> format = (WritablePixelFormat<ByteBuffer>) pixelReader.getPixelFormat();
        pixelReader.getPixels(0, 0, width, height, format, pixels, 0, width * bands);
        int[] shape = new int[] {height, width, bands};
        
        INDArray ret2 = Nd4j.create(1, pixels.length);
        for (int i = 0; i < ret2.length(); i += 4) {
            ret2.putScalar(i, ((int) pixels[i]) & 0xFF);
            ret2.putScalar(i, ((int) pixels[i + 1]) & 0xFF);
            ret2.putScalar(i, ((int) pixels[i + 2]) & 0xFF);
            ret2.putScalar(i, ((int) pixels[i + 3]) & 0xFF);
        }
        
        if (format.getType() == PixelFormat.Type.BYTE_BGRA_PRE || format.getType() == PixelFormat.Type.BYTE_BGRA) {
            for (int i = 0; i < ret2.length(); i += 4) {
                ret2.putScalar(i + 0, ((int) pixels[i + 3]) & 0xFF);
                ret2.putScalar(i + 1, ((int) pixels[i + 0]) & 0xFF);
                ret2.putScalar(i + 2, ((int) pixels[i + 1]) & 0xFF);
                ret2.putScalar(i + 3, ((int) pixels[i + 2]) & 0xFF);
            }
        } else {
            for (int i = 0; i < ret2.length(); i++) {
                ret2.putScalar(i, ((int) pixels[i]) & 0xFF);
            }
        }
            
        return ret2.reshape(shape).permute(2, 0, 1);
    }

    // TODO build flexibility on where to crop the image
    public Image centerCropIfNeeded(Image img) {
        int x = 0;
        int y = 0;
        int height = (int) img.getHeight();
        int width = (int) img.getWidth();
        int diff = Math.abs(width - height) / 2;

        if (width > height) {
            x = diff;
            width = width - diff;
        } else if (height > width) {
            y = diff;
            height = height - diff;
        }
        PixelReader pixelReader = img.getPixelReader();
        WritableImage writableImage = new WritableImage(width,  height);
        PixelWriter pixelWriter = writableImage.getPixelWriter();
        for (int i = x; i < width; i++) {
            for (int j = y; j < height; j++) {
                Color color = pixelReader.getColor(i, j);
                pixelWriter.setColor(i, j, color);
            }
        }
        return writableImage;
    }

    protected Image scalingIfNeed(Image image, boolean needAlpha) {
        return scalingIfNeed(image, height, width, needAlpha);
    }

    protected Image scalingIfNeed(Image image, int dstHeight, int dstWidth, boolean needAlpha) {
        if (dstHeight > 0 && dstWidth > 0 && (image.getHeight() != dstHeight || image.getWidth() != dstWidth)) {
            ImageView dummyImageView = new ImageView(image);
//            dummyImageView.setPreserveRatio(true);
            dummyImageView.setFitWidth(dstWidth);
            dummyImageView.setFitHeight(dstHeight);
            dummyImageView.setSmooth(true);
            return dummyImageView.snapshot(null, null);
        } else {
            return image;
        }
    }

    protected int getNumBands(PixelReader reader) {
        if (reader == null || reader.getPixelFormat() == null) {
            return 0;
        }
        switch (reader.getPixelFormat().getType()) {
            case INT_ARGB:
            case INT_ARGB_PRE:
            case BYTE_BGRA:
            case BYTE_BGRA_PRE: 
            case BYTE_INDEXED: return 4;
            default: return 3;
        }
    }
}
