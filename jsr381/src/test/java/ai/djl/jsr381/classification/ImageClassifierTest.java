/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package ai.djl.jsr381.classification;

import ai.djl.util.ZipUtils;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;
import java.util.Objects;
import javax.visrec.ml.classification.ImageClassifier;
import javax.visrec.ml.classification.NeuralNetImageClassifier;
import javax.visrec.ml.model.ModelCreationException;
import org.testng.annotations.Test;

public class ImageClassifierTest {

    @Test
    public void testImageClassifier() throws ModelCreationException {
        URL url = Objects.requireNonNull(ImageClassifierTest.class.getResource("/0.png"));
        Path input = Paths.get(url.getFile());

        Path modelDir = Paths.get("src/test/resources/mlp");

        ImageClassifier<BufferedImage> classifier =
                NeuralNetImageClassifier.builder()
                        .inputClass(BufferedImage.class)
                        .imageHeight(28)
                        .imageWidth(28)
                        .importModel(modelDir)
                        .build();

        Map<String, Float> result = classifier.classify(input);
        for (Map.Entry<String, Float> entry : result.entrySet()) {
            System.out.println(entry.getKey() + ": " + entry.getValue());
        }
    }

    @Test
    public void testImageClassifierTraining() throws IOException, ModelCreationException {
        Path trainingFile = downloadTrainingData();
        Path modelDir = Paths.get("build/model");

        ImageClassifier<BufferedImage> classifier =
                NeuralNetImageClassifier.builder()
                        .inputClass(BufferedImage.class)
                        .imageHeight(128)
                        .imageWidth(128)
                        .trainingFile(trainingFile)
                        .exportModel(modelDir)
                        .maxEpochs(2)
                        .build();

        Path input = trainingFile.resolve("cat/cat_1.png");
        Map<String, Float> result = classifier.classify(input);
        for (Map.Entry<String, Float> entry : result.entrySet()) {
            System.out.println(entry.getKey() + ": " + entry.getValue());
        }
    }

    private Path downloadTrainingData() throws IOException {
        String link =
                "https://github.com/JavaVisRec/jsr381-examples-datasets/raw/master/cats_and_dogs_training_data_png.zip";
        URL url = new URL(link);
        Path dir = Paths.get("datasets", "cats_and_dogs");
        if (!Files.exists(dir)) {
            Files.createDirectories(dir);
            try (InputStream is = url.openStream()) {
                ZipUtils.unzip(is, dir);
            }
        }
        return dir.resolve("training");
    }
}
