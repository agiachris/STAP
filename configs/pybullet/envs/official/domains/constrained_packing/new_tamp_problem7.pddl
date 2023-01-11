(define (problem new_tamp_problem7)
	(:domain workspace)
	(:objects
		blue_box - box
		cyan_box - box
		hook - hook
		rack - rack
		red_box - box
		yellow_box - box
	)
	(:init
		(on blue_box red_box)
		(on cyan_box rack)
		(on hook table)
		(on rack table)
		(on red_box rack)
		(on yellow_box rack)
	)
	(:goal
		(and
			(on rack table)
			(on hook table)
			(on blue_box red_box)
			(inhand red_box)
			(on yellow_box rack)
			(on cyan_box table)
		)
	)
)
