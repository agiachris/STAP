(define (problem new_tamp_problem6)
	(:domain workspace)
	(:objects
		blue_box - box
		cyan_box - box
		hook - tool
		rack - rack
		red_box - box
		yellow_box - box
	)
	(:init
		(on blue_box rack)
		(on cyan_box rack)
		(on hook table)
		(on rack table)
		(on red_box rack)
		(on yellow_box rack)
	)
	(:goal
		(and
			(on hook table)
			(on red_box rack)
			(on yellow_box table)
			(on blue_box rack)
			(on cyan_box rack)
			(on rack table)
		)
	)
)
