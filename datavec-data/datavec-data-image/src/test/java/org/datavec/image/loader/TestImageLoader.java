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

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Random;
import javafx.scene.image.Image;
import javafx.scene.image.PixelReader;
import javafx.scene.image.PixelWriter;
import javafx.scene.image.WritableImage;

import static org.junit.Assert.assertEquals;
import org.junit.Rule;


public class TestImageLoader {

    @Rule 
    public JavaFXThreadingRule javafxRule = new JavaFXThreadingRule();
    
    private static final long seed = 10;
    private static final Random rng = new Random(seed);

    @Test
    public void testToIntArrayArray() throws Exception {
        Image img = makeRandomImage(true);

        int w = (int) img.getWidth();
        int h = (int) img.getHeight();
        int ch = 4;
        ImageLoader loader = new ImageLoader(0, 0, ch);
        int[][] arr = loader.toIntArrayArray(img);

        assertEquals(h, arr.length);
        assertEquals(w, arr[0].length);

        PixelReader pixelReader = img.getPixelReader();
        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                assertEquals(pixelReader.getArgb(j, i), arr[i][j]);
            }
        }
    }

    @Test
    public void testToINDArrayBGR() throws Exception {
        Image img = makeRandomImage(false);
        int w = (int) img.getWidth();
        int h = (int) img.getHeight();
        int ch = 3;

        ImageLoader loader = new ImageLoader(0, 0, ch);
        INDArray arr = loader.toINDArrayBGR(img);

        int[] shape = arr.shape();
        assertEquals(3, shape.length);
        assertEquals(ch, shape[0]);
        assertEquals(h, shape[1]);
        assertEquals(w, shape[2]);

        PixelReader pixelReader = img.getPixelReader();
        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                int srcColor = pixelReader.getArgb(j, i);
                int a = 0xff << 24;
                int r = arr.getInt(2, i, j) << 16;
                int g = arr.getInt(1, i, j) << 8;
                int b = arr.getInt(0, i, j) & 0xff;
                int dstColor = a | r | g | b;
                assertEquals(srcColor, dstColor);
            }
        }
    }

    @Test
    public void testScalingIfNeed() throws Exception {
        Image img1 = makeRandomImage(true);
        Image img2 = makeRandomImage(false);

        int w1 = 60, h1 = 110, ch1 = 6;
        ImageLoader loader1 = new ImageLoader(h1, w1, ch1);

        Image scaled1 = loader1.scalingIfNeed(img1, true);
        assertEquals(w1, scaled1.getWidth(), 0);
        assertEquals(h1, scaled1.getHeight(), 0);
        assertEquals(4, loader1.getNumBands(scaled1.getPixelReader()));

        Image scaled2 = loader1.scalingIfNeed(img1, false);
        assertEquals(w1, scaled2.getWidth(), 0);
        assertEquals(h1, scaled2.getHeight(), 0);
        assertEquals(4, loader1.getNumBands(scaled2.getPixelReader()));

        Image scaled3 = loader1.scalingIfNeed(img2, true);
        assertEquals(w1, scaled3.getWidth(), 0);
        assertEquals(h1, scaled3.getHeight(), 0);
        assertEquals(4, loader1.getNumBands(scaled3.getPixelReader()));

        Image scaled4 = loader1.scalingIfNeed(img2, false);
        assertEquals(w1, scaled4.getWidth(), 0);
        assertEquals(h1, scaled4.getHeight(), 0);
        assertEquals(4, loader1.getNumBands(scaled4.getPixelReader()));

        int w2 = 70, h2 = 120, ch2 = 6;
        ImageLoader loader2 = new ImageLoader(h2, w2, ch2);

        Image scaled5 = loader2.scalingIfNeed(img1, true);
        assertEquals(w2, scaled5.getWidth(), 0);
        assertEquals(h2, scaled5.getHeight(), 0);
        assertEquals(4, loader2.getNumBands(scaled5.getPixelReader()));

        Image scaled6 = loader2.scalingIfNeed(img1, false);
        assertEquals(w2, scaled6.getWidth(), 0);
        assertEquals(h2, scaled6.getHeight(), 0);
        assertEquals(4, loader2.getNumBands(scaled6.getPixelReader()));

    }

    private Image makeRandomImage(boolean alpha) {
        int w = rng.nextInt() % 100 + 100;
        int h = rng.nextInt() % 100 + 100;
        WritableImage writableImage = new WritableImage(w, h);
        PixelWriter pixelWriter = writableImage.getPixelWriter();
        
        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                int a = (alpha ? rng.nextInt() : 1) & 0xff;
                int r = rng.nextInt() & 0xff;
                int g = rng.nextInt() & 0xff;
                int b = rng.nextInt() & 0xff;
                int v = (a << 24) | (r << 16) | (g << 8) | b;
                pixelWriter.setArgb(j, i, v);
            }
        }
        return writableImage;
    }
}
