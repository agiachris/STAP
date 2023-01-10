(define (problem new_tamp_problem0)
	(:domain workspace)
	(:objects
		cyan_box - box
		hook - tool
		rack - rack
		red_box - box
	)
	(:init
		(on cyan_box rack)
		(on hook table)
		(on rack table)
		(on red_box rack)
	)
	(:goal
		(and
			(on hook table)
			(on red_box table)
			(on cyan_box table)
			(on rack table)
		)
	)
)
