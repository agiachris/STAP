(define (problem new_tamp_problem3)
	(:domain workspace)
	(:objects
		blue_box - box
		hook - tool
		rack - rack
	)
	(:init
		(on blue_box rack)
		(on hook table)
		(on rack table)
	)
	(:goal
		(and
			(on hook rack)
			(on blue_box rack)
			(on rack table)
		)
	)
)
