package com.moji.ai.controller;

import com.moji.ai.dto.ConsensusPrediction;
import com.moji.ai.dto.PredictionResponse;
import com.moji.ai.dto.ProjectRequest;
import com.moji.ai.service.ConsensusMLService;
import com.moji.ai.service.MLPredictionService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/projects")
public class ProjectController {

    private final MLPredictionService mlService;
    private final ConsensusMLService consensusService;

    public ProjectController(MLPredictionService mlService, ConsensusMLService consensusService) {
        this.mlService = mlService;
        this.consensusService = consensusService;
    }

    @PostMapping("/predict")
    public ResponseEntity<PredictionResponse> predictProject(
            @RequestParam int teamSize,
            @RequestParam int complexityScore) {

        try {
            PredictionResponse prediction = mlService.predictProject(teamSize, complexityScore);
            return ResponseEntity.ok(prediction);
        } catch (Exception e) {
            return ResponseEntity.status(503)
                    .body(new PredictionResponse(-1, 0.0, "ML service unavailable"));
        }
    }

    // Alternative JSON body endpoint
    @PostMapping("/predict-json")
    public ResponseEntity<PredictionResponse> predictProjectJson(
            @RequestBody ProjectRequest request) {

        try {
            PredictionResponse prediction = mlService.predictProject(
                    request.team_size(), request.complexity_score());
            return ResponseEntity.ok(prediction);
        } catch (Exception e) {
            return ResponseEntity.status(503)
                    .body(new PredictionResponse(-1, 0.0, "ML service unavailable"));
        }
    }

    @GetMapping("/ml-health")
    public ResponseEntity<String> checkMLHealth() {
        boolean healthy = mlService.isMLServiceHealthy();
        return healthy ?
                ResponseEntity.ok("ML service is healthy") :
                ResponseEntity.status(503).body("ML service is down");
    }

    // Add this to your ProjectController.java
    @PostMapping("/consensus-predict")
    public ResponseEntity<ConsensusPrediction> consensusPredict(
            @RequestParam int teamSize,
            @RequestParam int complexityScore) {

        ConsensusPrediction result = consensusService.getConsensusPrediction(teamSize, complexityScore);
        return ResponseEntity.ok(result);
    }
}
