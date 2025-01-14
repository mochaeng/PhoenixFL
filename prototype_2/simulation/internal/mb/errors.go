package mb

import "errors"

var (
	ErrInvalidChannel        = errors.New("invalid channel")
	ErrPublishConfirmTimeout = errors.New("timeout waiting for message confirmation")
	ErrNackedMessage         = errors.New("message was not acknowledged by rabbitMQ")
)
