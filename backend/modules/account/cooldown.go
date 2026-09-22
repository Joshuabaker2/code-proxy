package account

import (
	"math"
	"time"
)

// backoffDecayWindow is how long after a cooldown expires the accumulated
// backoff level is still held against an account. Without it the level only
// ever grows between successes, so a long-lived row eventually sits at the
// exponential cap permanently.
const backoffDecayWindow = 10 * time.Minute

// CooldownForStatus returns the cooldown duration for an HTTP status code.
//
// Cooldown exists to take an *account* out of rotation, so only account-scoped
// failures belong here. A zero duration means the failure says nothing about
// this account's health and must not make it unselectable: upstream 5xx and
// 529 overload are the provider having a bad moment, and 4xx statuses such as
// 400 "prompt is too long" or 404 "unknown model" are properties of the
// request. Cooling the account down for those strands a single-account setup
// and makes the next Select look like "no account configured".
func CooldownForStatus(httpStatus int, backoffLevel int) time.Duration {
	switch httpStatus {
	case 401:
		return 5 * time.Minute // Invalid token
	case 402:
		return 30 * time.Minute // Payment required
	case 403:
		return 30 * time.Minute // Forbidden
	case 429:
		return ExponentialBackoff(backoffLevel) // Rate limited
	default:
		return 0
	}
}

// ExponentialBackoff calculates the wait time: 1s x 2^level, max 2min
func ExponentialBackoff(level int) time.Duration {
	if level < 0 {
		level = 0
	}
	d := time.Duration(math.Pow(2, float64(level))) * time.Second
	if d > 2*time.Minute {
		d = 2 * time.Minute
	}
	return d
}

// decayBackoffLevel drops the stored level back to zero once the previous
// cooldown has been expired for longer than backoffDecayWindow, so unrelated
// failures spread over hours do not compound.
func decayBackoffLevel(level int, cooldownUntil *time.Time, now time.Time) int {
	if level <= 0 {
		return 0
	}
	if cooldownUntil == nil || now.Sub(*cooldownUntil) > backoffDecayWindow {
		return 0
	}
	return level
}
