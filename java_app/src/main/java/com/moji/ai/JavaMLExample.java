package com.moji.ai;

import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import java.util.ArrayList;

public class JavaMLExample {

    public static void main(String[] args) throws Exception {
        // Create dataset structure
        ArrayList<Attribute> attributes = new ArrayList<>();
        attributes.add(new Attribute("team_size"));
        attributes.add(new Attribute("complexity_score"));

        ArrayList<String> classValues = new ArrayList<>();
        classValues.add("delayed");
        classValues.add("on_time");
        attributes.add(new Attribute("on_time", classValues));

        Instances dataset = new Instances("ProjectData", attributes, 0);
        dataset.setClassIndex(dataset.numAttributes() - 1);

        // Add training data
        double[][] trainingData = {
                {3, 2, 1}, {5, 7, 0}, {8, 9, 0}, {4, 3, 1}, {6, 6, 0},
                {7, 8, 0}, {3, 1, 1}, {9, 10, 0}, {5, 4, 1}, {8, 7, 0}
        };

        for (double[] row : trainingData) {
            Instance instance = new DenseInstance(1.0, row);
            dataset.add(instance);
        }

        // Train model
        RandomForest classifier = new RandomForest();
        classifier.setNumIterations(10);
        classifier.buildClassifier(dataset);

        // Make prediction
        Instance testInstance = new DenseInstance(1.0, new double[]{6, 5, 0});
        testInstance.setDataset(dataset);

        double prediction = classifier.classifyInstance(testInstance);
        double[] probabilities = classifier.distributionForInstance(testInstance);

        String result = prediction == 0 ? "DELAYED" : "ON TIME";
        System.out.printf("Prediction for team_size=6, complexity=5: %s%n", result);
        System.out.printf("Confidence: %.2f%n", probabilities[(int)prediction]);
    }
}