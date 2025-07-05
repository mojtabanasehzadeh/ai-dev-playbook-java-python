package com.moji.ai.dto;

// JavaMLResult.java - Add this too
public record JavaMLResult(
        int prediction,
        double probability,
        String message
) {
}