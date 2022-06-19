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

import java.awt.image.BufferedImage;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;

import javax.visrec.ml.classification.ImageClassifier;
import javax.visrec.ml.classification.NeuralNetImageClassifier;
import javax.visrec.ml.model.ModelCreationException;

public class ImageClassifierExample {

    public static void main(String[] args) throws ModelCreationException {
        Path input = Paths.get("../jsr381/src/test/resources/0.png");

        // use pre-trained mlp model
        Path modelDir = Paths.get("../jsr381/src/test/resources/mlp");

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
}
