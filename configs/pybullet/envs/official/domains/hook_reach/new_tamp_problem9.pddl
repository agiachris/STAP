(define (problem new_tamp_problem9)
	(:domain workspace)
	(:objects
		cyan_box - box
		hook - tool
		rack - rack
	)
	(:init
		(on cyan_box rack)
		(on hook table)
		(on rack table)
	)
	(:goal
		(and
			(on hook table)
			(on rack table)
			(on cyan_box rack)
		)
	)
)
