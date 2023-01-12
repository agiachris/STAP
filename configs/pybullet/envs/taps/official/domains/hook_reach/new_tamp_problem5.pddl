(define (problem new_tamp_problem5)
	(:domain workspace)
	(:objects
		cyan_box - box
		hook - tool
		rack - rack
		red_box - box
		yellow_box - box
	)
	(:init
		(on cyan_box rack)
		(on hook table)
		(on rack table)
		(on red_box rack)
		(on yellow_box rack)
	)
	(:goal
		(and
			(on hook table)
			(on yellow_box table)
			(on cyan_box table)
			(on red_box table)
			(on rack table)
		)
	)
)
