(define (problem new_tamp_problem0)
	(:domain workspace)
	(:objects
		blue_box - box
		cyan_box - box
		hook - hook
		rack - rack
		yellow_box - box
	)
	(:init
		(on blue_box yellow_box)
		(on cyan_box table)
		(on hook table)
		(on rack table)
		(on yellow_box hook)
	)
	(:goal
		(and
			(on rack table)
			(on hook table)
			(on yellow_box hook)
			(inhand blue_box)
			(on cyan_box table)
		)
	)
)
