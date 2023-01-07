(define (problem new_tamp_problem2)
	(:domain workspace)
	(:objects
		cyan_box - box
		hook - tool
		rack - rack
		yellow_box - box
	)
	(:init
		(on cyan_box rack)
		(on hook table)
		(on rack table)
		(on yellow_box rack)
	)
	(:goal
		(and
			(on hook rack)
			(on rack table)
			(on yellow_box table)
			(on cyan_box rack)
		)
	)
)
