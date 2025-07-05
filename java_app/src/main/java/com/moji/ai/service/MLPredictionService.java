package com.moji.ai.service;

import com.moji.ai.dto.PredictionResponse;
import com.moji.ai.dto.ProjectRequest;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

// Service to call Python ML API
@Service
public class MLPredictionService {

    private final WebClient webClient;

    public MLPredictionService(WebClient.Builder webClientBuilder) {
        String ML_SERVICE_URL = "http://localhost:8000";
        this.webClient = webClientBuilder.baseUrl(ML_SERVICE_URL).build();
    }

    public PredictionResponse predictProject(int teamSize, int complexityScore) {
        ProjectRequest request = new ProjectRequest(teamSize, complexityScore);

        return webClient.post()
                .uri("/predict")
                .bodyValue(request)
                .retrieve()
                .bodyToMono(PredictionResponse.class)
                .block(); // For synchronous call - in production use reactive approach
    }

    // Reactive version (better for production)
    public Mono<PredictionResponse> predictProjectAsync(int teamSize, int complexityScore) {
        ProjectRequest request = new ProjectRequest(teamSize, complexityScore);

        return webClient.post()
                .uri("/predict")
                .bodyValue(request)
                .retrieve()
                .bodyToMono(PredictionResponse.class);
    }

    public boolean isMLServiceHealthy() {
        try {
            return webClient.get()
                    .uri("/health")
                    .retrieve()
                    .bodyToMono(String.class)
                    .block() != null;
        } catch (Exception e) {
            return false;
        }
    }
}