//! OpenAI-compatible error response type.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;

pub struct AppError {
    pub status: StatusCode,
    pub message: String,
    pub error_type: &'static str,
}

impl AppError {
    pub fn bad_request(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: msg.into(),
            error_type: "invalid_request_error",
        }
    }

    pub fn unauthorized(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            message: msg.into(),
            error_type: "authentication_error",
        }
    }

    pub fn internal(err: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: err.to_string(),
            error_type: "server_error",
        }
    }

    pub fn payload_too_large(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::PAYLOAD_TOO_LARGE,
            message: msg.into(),
            error_type: "invalid_request_error",
        }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let body = Json(json!({
            "error": {
                "message": self.message,
                "type": self.error_type,
                "param": null,
                "code": null
            }
        }));
        (self.status, body).into_response()
    }
}
