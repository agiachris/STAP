(define (problem new_tamp_problem1)
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
			(on hook rack)
			(on cyan_box table)
			(on rack table)
		)
	)
)
