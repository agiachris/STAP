(define (problem new_tamp_problem6)
	(:domain workspace)
	(:objects
		hook - hook
		rack - rack
		red_box - box
		yellow_box - box
	)
	(:init
		(on hook table)
		(on rack table)
		(on red_box table)
		(on yellow_box rack)
	)
	(:goal
		(and
			(on yellow_box table)
			(on rack table)
			(inhand red_box)
			(on hook table)
		)
	)
)
