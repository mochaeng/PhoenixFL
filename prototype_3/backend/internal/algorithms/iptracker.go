package algorithms

import "github.com/mochaeng/phoenixfl/internal/models"

type IpTracker interface {
	AddOrUpdateIpCount(address string)
	GetTopIps(n int) []*models.IpCount
}
