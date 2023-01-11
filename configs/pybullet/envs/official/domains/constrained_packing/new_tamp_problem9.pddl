(define (problem new_tamp_problem9)
	(:domain workspace)
	(:objects
		hook - hook
		rack - rack
		red_box - box
	)
	(:init
		(on hook table)
		(on rack table)
		(on red_box rack)
	)
	(:goal
		(and
			(on hook table)
			(on rack table)
			(inhand red_box)
		)
	)
)
