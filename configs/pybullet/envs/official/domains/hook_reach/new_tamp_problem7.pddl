(define (problem new_tamp_problem7)
	(:domain workspace)
	(:objects
		blue_box - box
		hook - tool
		rack - rack
		red_box - box
	)
	(:init
		(on blue_box rack)
		(on hook table)
		(on rack table)
		(on red_box rack)
	)
	(:goal
		(and
			(on hook table)
			(on red_box rack)
			(on blue_box rack)
			(on rack table)
		)
	)
)
