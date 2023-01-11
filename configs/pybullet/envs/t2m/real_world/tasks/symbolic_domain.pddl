(define (domain symbolic_workspace)
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
		(beyondworkspace ?a - physobj)
		(under ?a - movable ?b - receptacle)
	)
	(:action pick
		:parameters (?a - movable)
		:precondition (and
			(exists (?b - unmovable) (on ?a ?b))
			(forall (?b - movable) (not (inhand ?b)))
		)
		:effect (and
			(inhand ?a)
			(forall (?b - unmovable) (not (on ?a ?b)))
		)
	)
	(:action place
		:parameters (?a - movable ?b - unmovable)
		:precondition (and
			(not (= ?a ?b))
			(inhand ?a)
		)
		:effect (and
			(not (inhand ?a))
			(on ?a ?b)
		)
	)
	(:action pull
		:parameters (?a - box ?b - tool)
		:precondition (and
			(not (= ?a ?b))
			(inhand ?b)
			(on ?a table)
		)
		:effect (and
			(inworkspace ?a)
		)
	)
    (:action push
        :parameters (?a - box ?b - tool ?c - receptacle)
        :precondition (and
            (inhand ?b)
            (on ?a table)
            (not (under ?a ?c))
        )
        :effect (and
            (under ?a ?c)
			(beyondworkspace ?a)
        )
    )
)
