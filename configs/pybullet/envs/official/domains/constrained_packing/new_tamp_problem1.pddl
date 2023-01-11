(define (problem new_tamp_problem1)
	(:domain workspace)
	(:objects
		cyan_box - box
		hook - hook
		rack - rack
		red_box - box
		yellow_box - box
	)
	(:init
		(on cyan_box rack)
		(on hook table)
		(on rack table)
		(on red_box hook)
		(on yellow_box hook)
	)
	(:goal
		(and
			(on rack table)
			(inhand cyan_box)
			(on hook table)
			(on red_box hook)
			(on yellow_box hook)
		)
	)
)
