"""Python port of nfl4th (Ben Baldwin and nflverse contributors, MIT license).

Component models (conversion, field goal, punt) are ported faithfully from the R
package. Post-play states are meant to be valued by a Kalshi market-response model;
`nfl4th_py.models.wp_baseline` reproduces the original WP-based valuation and exists
only as a benchmark.
"""

__version__ = "0.1.0"
