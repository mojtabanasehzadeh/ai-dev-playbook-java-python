package com.moji.ai.service;

import com.moji.ai.dto.ConsensusPrediction;
import com.moji.ai.dto.JavaMLResult;
import com.moji.ai.dto.PredictionResponse;
import org.springframework.stereotype.Service;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;

@Service
public class ConsensusMLService {

    private final MLPredictionService pythonService;
    private final RandomForest javaModel;
    private final Instances javaDataset;

    public ConsensusMLService(MLPredictionService pythonService) throws Exception {
        this.pythonService = pythonService;
        this.javaDataset = createDataset();
        this.javaModel = initializeJavaModel();
    }

    public ConsensusPrediction getConsensusPrediction(int teamSize, int complexityScore) {
        try {
            // Get Python prediction
            PredictionResponse pythonResult = pythonService.predictProject(teamSize, complexityScore);

            // Get Java prediction
            JavaMLResult javaResult = getJavaPrediction(teamSize, complexityScore);

            // Calculate consensus
            return calculateConsensus(pythonResult, javaResult, teamSize, complexityScore);

        } catch (Exception e) {
            return new ConsensusPrediction(-1, 0.0, "Error in consensus calculation",
                    null, null, "ERROR: " + e.getMessage());
        }
    }

    private JavaMLResult getJavaPrediction(int teamSize, int complexityScore) throws Exception {
        Instance testInstance = new DenseInstance(1.0, new double[]{teamSize, complexityScore, 0});
        testInstance.setDataset(javaDataset);

        double prediction = javaModel.classifyInstance(testInstance);
        double[] probabilities = javaModel.distributionForInstance(testInstance);

        String message = prediction == 0 ? "Project likely DELAYED" : "Project likely ON TIME";

        return new JavaMLResult((int) prediction, probabilities[(int) prediction], message);
    }

    private ConsensusPrediction calculateConsensus(PredictionResponse pythonResult,
                                                   JavaMLResult javaResult,
                                                   int teamSize, int complexityScore) {

        // Simple voting approach
        int finalPrediction;
        double consensusConfidence;
        String consensusReason;

        if (pythonResult.prediction() == javaResult.prediction()) {
            // Both models agree
            finalPrediction = pythonResult.prediction();
            consensusConfidence = Math.max(pythonResult.probability(), javaResult.probability());
            consensusReason = "AGREEMENT - Both models predict the same outcome";
        } else {
            // Models disagree - use the more confident one
            if (pythonResult.probability() > javaResult.probability()) {
                finalPrediction = pythonResult.prediction();
                consensusConfidence = pythonResult.probability() * 0.7; // Lower confidence due to disagreement
                consensusReason = "DISAGREEMENT - Using Python model (higher confidence)";
            } else {
                finalPrediction = javaResult.prediction();
                consensusConfidence = javaResult.probability() * 0.7; // Lower confidence due to disagreement
                consensusReason = "DISAGREEMENT - Using Java model (higher confidence)";
            }
        }

        // Add domain knowledge insight
        String domainInsight = getDomainInsight(teamSize, complexityScore);
        String fullReasoning = consensusReason + " | " + domainInsight;

        String message = finalPrediction == 1 ? "Project likely ON TIME" : "Project likely DELAYED";

        return new ConsensusPrediction(
                finalPrediction,
                Math.round(consensusConfidence * 100.0) / 100.0,
                message,
                pythonResult,
                javaResult,
                fullReasoning
        );
    }

    private String getDomainInsight(int teamSize, int complexityScore) {
        // Your 15 years of PM experience encoded as rules

        if (teamSize <= 3 && complexityScore <= 3) {
            return "Small team, simple project - typically manageable";
        }

        if (teamSize >= 8 && complexityScore >= 8) {
            return "Large team, complex project - high coordination risk";
        }

        if (teamSize <= 4 && complexityScore >= 7) {
            return "Small team, high complexity - potential resource constraint";
        }

        if (teamSize >= 7 && complexityScore <= 4) {
            return "Large team, simple project - potential over-engineering";
        }

        double ratio = (double) complexityScore / teamSize;
        if (ratio > 1.5) {
            return "High complexity-to-team ratio - monitor closely";
        }

        if (ratio < 0.7) {
            return "Low complexity-to-team ratio - good resource allocation";
        }

        return "Standard project parameters - outcome depends on execution";
    }

    private RandomForest initializeJavaModel() throws Exception {
        // Same training data as in JavaMLExample
        double[][] trainingData = {
                {3, 2, 1}, {5, 7, 0}, {8, 9, 0}, {4, 3, 1}, {6, 6, 0},
                {7, 8, 0}, {3, 1, 1}, {9, 10, 0}, {5, 4, 1}, {8, 7, 0},
                {2, 1, 1}, {10, 9, 0}, {4, 5, 1}, {7, 8, 0}, {6, 6, 0}
        };

        for (double[] row : trainingData) {
            Instance instance = new DenseInstance(1.0, row);
            javaDataset.add(instance);
        }

        RandomForest classifier = new RandomForest();
        classifier.setNumIterations(100);
        classifier.setSeed(42);
        classifier.buildClassifier(javaDataset);

        return classifier;
    }

    private Instances createDataset() {
        ArrayList<Attribute> attributes = new ArrayList<>();
        attributes.add(new Attribute("team_size"));
        attributes.add(new Attribute("complexity_score"));

        ArrayList<String> classValues = new ArrayList<>();
        classValues.add("delayed");    // 0
        classValues.add("on_time");    // 1
        attributes.add(new Attribute("on_time", classValues));

        Instances dataset = new Instances("ProjectData", attributes, 0);
        dataset.setClassIndex(dataset.numAttributes() - 1);

        return dataset;
    }
}