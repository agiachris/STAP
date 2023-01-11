(define (problem new_tamp_problem5)
	(:domain workspace)
	(:objects
		blue_box - box
		hook - hook
		rack - rack
		red_box - box
	)
	(:init
		(on blue_box rack)
		(on hook table)
		(on rack table)
		(on red_box table)
	)
	(:goal
		(and
			(on blue_box table)
			(on hook table)
			(on rack table)
			(on red_box table)
		)
	)
)
