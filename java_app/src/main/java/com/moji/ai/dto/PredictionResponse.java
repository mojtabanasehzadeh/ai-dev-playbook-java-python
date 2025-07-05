package com.moji.ai.dto;

public record PredictionResponse(int prediction, double probability, String message) {
}

