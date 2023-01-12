(define (domain geometric_workspace)
	(:requirements :strips :typing :equality :universal-preconditions :negative-preconditions :conditional-effects)
	(:types
		physobj - object
		unmovable - physobj
		movable - physobj
		tool - movable
		box - movable
		receptacle - unmovable
	)
	(:constants table - unmovable)
	(:predicates
		(inhand ?a - movable)
		(on ?a - movable ?b - unmovable)
		(inworkspace ?a - physobj)
		(incollisionzone ?a - physobj)
		(inoperationalzone ?a - physobj)
		(inobstructionzone ?a - physobj)
		(beyondworkspace ?a - physobj)
		(inoodzone ?a - physobj)
		(infront ?a - physobj ?b - physobj)
		(nonblocking ?a - physobj ?b - physobj)
		(under ?a - movable ?b - receptacle)
	)
	(:action pick
		:parameters (?a - movable)
		:precondition (and
			; Modify in delete list
			(forall (?b - movable) (not (inhand ?b)))
			(exists (?b - unmovable) 
				(and (on ?a ?b)
					 (inworkspace ?b)))
			(or (inworkspace ?a)
				(incollisionzone ?a)
				(inoperationalzone ?a)
				(inobstructionzone ?a))
			; Additional checks
			(forall (?b - receptacle) (not (under ?a ?b)))
			(not (inoodzone ?a))
			(not (beyondworkspace ?a))	
		)
		:effect (and
			(inhand ?a)
			(forall (?b - unmovable) (not (on ?a ?b)))
			(when (and (inworkspace ?a))
				  (and (not (inworkspace ?a))))
			(when (and (incollisionzone ?a))
				  (and (not (incollisionzone ?a))))
			(when (and (inoperationalzone ?a))
				  (and (not (inoperationalzone ?a))))
			(when (and (inobstructionzone ?a))
				  (and (not (inobstructionzone ?a))))
			(forall (?b - physobj)
				(and (not (infront ?a ?b))
					 (not (infront ?b ?a))
					 (not (nonblocking ?a ?b))
					 (not (nonblocking ?b ?a))))
		)
	)
	(:action place
		:parameters (?a - movable ?b - unmovable)
		:precondition (and
			; Modify in delete list
			(inhand ?a)
			; Additional checks
			(inworkspace ?b)
			(not (= ?a ?b))
		)
		:effect (and
			(not (inhand ?a))
			(on ?a ?b)
			(inworkspace ?a)
			(forall (?c - receptacle)
				(when (and (= ?b table)
						   (beyondworkspace ?c))
					  (and (inoperationalzone ?a)
					  	   (infront ?a ?c))))
		)
	)
	(:action pull
		:parameters (?a - box ?b - tool)
		:precondition (and
			; Modify in delete list
			(not (inworkspace ?a))
			(beyondworkspace ?a)
			; Additional checks
			(inhand ?b)
			(on ?a table)
			(forall (?c - physobj) (not (infront ?c ?a)))
			(forall (?c - receptacle) (not (under ?a ?c)))
			(not (incollisionzone ?a))
			(not (inoperationalzone ?a))
			(not (inobstructionzone ?a))
			(not (inoodzone ?a))
			(not (= ?a ?b))
		)
		:effect (and
			(inworkspace ?a)
			(not (beyondworkspace ?a))
		)
	)
	(:action push
        :parameters (?a - box ?b - tool ?c - receptacle)
        :precondition (and
			; Modify in delete list
			(inoperationalzone ?a)
			(infront ?a ?c)
			; Additional checks
            (inhand ?b)
            (on ?a table)
			(beyondworkspace ?c)
			(not (incollisionzone ?a))
			(not (inobstructionzone ?a))
			(not (beyondworkspace ?a))
			(not (inoodzone ?a))
			(forall (?d - receptacle) (not (under ?a ?d)))
			(forall (?d - movable)
				(or (= ?a ?d)
					(and (not (= ?a ?d))
						 (not (inobstructionzone ?d)))))
        )
        :effect (and
			(beyondworkspace ?a)
            (under ?a ?c)
			(not (infront ?a ?c))
			(when (and (inworkspace ?a))
				  (and (not (inworkspace ?a))))
			(when (and (inoperationalzone ?a))
				  (and (not (inoperationalzone ?a))))
        )
    )
)
