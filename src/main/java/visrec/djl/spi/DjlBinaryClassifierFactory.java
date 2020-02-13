package visrec.djl.spi;

import ai.djl.Model;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.BatchNorm;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.evaluator.BinaryAccuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import java.io.IOException;
import javax.visrec.ml.ClassifierCreationException;
import javax.visrec.ml.classification.BinaryClassifier;
import javax.visrec.ml.classification.NeuralNetBinaryClassifier;
import javax.visrec.spi.BinaryClassifierFactory;
import visrec.djl.ml.classification.SimpleBinaryClassifier;
import visrec.djl.ml.dataset.CsvDataset;

public class DjlBinaryClassifierFactory implements BinaryClassifierFactory<float[]> {

    @Override
    public Class<float[]> getTargetClass() {
        return float[].class;
    }

    @Override
    public BinaryClassifier<float[]> create(NeuralNetBinaryClassifier.BuildingBlock<float[]> block)
            throws ClassifierCreationException {
        int inputSize = block.getInputsNum();
        int[] hiddenLayers = block.getHiddenLayers();
        int epochs = block.getMaxEpochs();
        int batchSize = 32;

        SequentialBlock mlp = new SequentialBlock().add(Blocks.batchFlattenBlock(inputSize));
        for (int size : hiddenLayers) {
            mlp.add(new Linear.Builder().setOutChannels(size).build()).add(Activation::relu);
        }
        mlp.add(new BatchNorm.Builder().build())
                .add(new Linear.Builder().setOutChannels(1).build())
                .add(arrays -> new NDList(arrays.singletonOrThrow().flatten()));

        Model model = Model.newInstance();
        model.setBlock(mlp);

        RandomAccessDataset[] dataset;
        try {
            CsvDataset csv =
                    CsvDataset.builder()
                            .setCsvFile(block.getTrainingFile())
                            .setSampling(batchSize, true)
                            .build();
            dataset = csv.randomSplit(8, 2);
        } catch (IOException e) {
            throw new ClassifierCreationException("Failed to load dataset.", e);
        }

        // setup training configuration
        DefaultTrainingConfig config =
                new DefaultTrainingConfig(Loss.sigmoidBinaryCrossEntropyLoss())
                        .addTrainingListeners(
                                TrainingListener.Defaults.logging(
                                        BinaryClassifier.class.getSimpleName(),
                                        batchSize,
                                        (int) dataset[0].getNumIterations(),
                                        (int) dataset[1].getNumIterations(),
                                        null))
                        .addEvaluator(new BinaryAccuracy());

        try (Trainer trainer = model.newTrainer(config)) {
            trainer.setMetrics(new Metrics());
            Shape inputShape = new Shape(1, inputSize);
            trainer.initialize(inputShape);

            for (int i = 0; i < epochs; i++) {
                for (Batch batch : trainer.iterateDataset(dataset[0])) {
                    trainer.trainBatch(batch);
                    trainer.step();
                    batch.close();
                }

                for (Batch batch : trainer.iterateDataset(dataset[1])) {
                    trainer.validateBatch(batch);
                    batch.close();
                }

                // reset training and validation evaluators at end of epoch
                trainer.endEpoch();
            }
        }

        return new SimpleBinaryClassifier(new ZooModel<>(model, new BinaryClassifierTranslator()));
    }

    private static final class BinaryClassifierTranslator implements Translator<float[], Float> {

        @Override
        public NDList processInput(TranslatorContext ctx, float[] input) {
            NDManager manager = ctx.getNDManager();
            NDArray array = manager.create(input);
            return new NDList(array);
        }

        @Override
        public Float processOutput(TranslatorContext ctx, NDList list) {
            return list.singletonOrThrow().getFloat();
        }
    }
}
