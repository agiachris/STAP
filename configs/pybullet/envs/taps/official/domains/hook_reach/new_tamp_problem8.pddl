(define (problem new_tamp_problem8)
	(:domain workspace)
	(:objects
		hook - tool
		rack - rack
		yellow_box - box
	)
	(:init
		(on hook table)
		(on rack table)
		(on yellow_box rack)
	)
	(:goal
		(and
			(on hook rack)
			(on yellow_box rack)
			(on rack table)
		)
	)
)
