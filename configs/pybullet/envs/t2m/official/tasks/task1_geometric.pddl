(define (problem lh_1)
	(:domain geometric_workspace)
	(:objects
        rack - receptacle
		hook - tool
        cyan_box - box
		yellow_box - box
        blue_box - box
	)
	(:init
		(on rack table)
		(on hook table)
		(on yellow_box table)
        (on blue_box table)
        (on cyan_box table)
        ; Geometric facts
        (inworkspace table)
        (inworkspace hook)
        (inoperationalzone cyan_box)
        (inobstructionzone yellow_box)
        (inobstructionzone blue_box)
        (beyondworkspace rack)
        (infront cyan_box rack)
        (infront yellow_box rack)
        (infront blue_box rack)
        (nonblocking rack hook)
	)
	(:goal (and
		(under cyan_box rack)
	))
)
