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

import java.nio.file.Path;
import java.nio.file.Paths;
import javax.visrec.ml.classification.BinaryClassifier;
import javax.visrec.ml.classification.NeuralNetBinaryClassifier;
import javax.visrec.ml.model.ModelCreationException;

public class BinaryClassifierExample {

    public static void main(String[] args) throws ModelCreationException {
        Path trainingFile = Paths.get("../jsr381/src/test/resources/spam.csv");
        BinaryClassifier<float[]> spamClassifier =
                NeuralNetBinaryClassifier.builder()
                        .inputClass(float[].class)
                        .inputsNum(57)
                        .hiddenLayers(5)
                        .maxEpochs(2)
                        .trainingPath(trainingFile)
                        .build();

        // create test email feature
        float[] emailFeatures = new float[57];
        emailFeatures[56] = 1;

        Float result = spamClassifier.classify(emailFeatures);
        System.out.println(result);
    }
}
