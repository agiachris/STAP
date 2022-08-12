(define (problem pick-box)
	(:domain workspace)
	(:objects
		hook - movable
		a - movable
	)
	(:init
		(on hook table)
		(on a table)
	)
	(:goal (and
		(inhand a)
	))
)
