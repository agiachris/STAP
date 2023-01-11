(define (problem new_tamp_problem2)
	(:domain workspace)
	(:objects
		blue_box - box
		cyan_box - box
		hook - hook
		rack - rack
	)
	(:init
		(on blue_box rack)
		(on cyan_box blue_box)
		(on hook table)
		(on rack table)
	)
	(:goal
		(and
			(on hook table)
			(on rack table)
			(on cyan_box table)
			(on blue_box table)
		)
	)
)
