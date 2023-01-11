(define (problem new_tamp_problem4)
	(:domain workspace)
	(:objects
		blue_box - box
		cyan_box - box
		hook - hook
		rack - rack
		yellow_box - box
	)
	(:init
		(on blue_box rack)
		(on cyan_box rack)
		(on hook table)
		(on rack table)
		(on yellow_box rack)
	)
	(:goal
		(and
			(on rack table)
			(on hook table)
			(on blue_box table)
			(on yellow_box rack)
			(on cyan_box table)
		)
	)
)
